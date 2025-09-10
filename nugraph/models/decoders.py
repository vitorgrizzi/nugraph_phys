from typing import Any, Callable

from abc import ABC

import torch
from torch import Tensor, tensor, cat
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.aggr import SoftmaxAggregation, LSTMAggregation, SumAggregation, MeanAggregation, \
    MaxAggregation, MinAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver
from torch_geometric.nn import GlobalAttention
import torchmetrics as tm

import matplotlib.pyplot as plt
import seaborn as sn

from ..util import RecallLoss, LogCoshLoss, ObjCondensationLoss, MichelLoss, KLDivergenceLoss, \
    CrossEntropyDistributionLoss, BalancedFocalRecallLoss


class DecoderBase(nn.Module, ABC):
    '''Base class for all NuGraph decoders'''

    def __init__(self,
                 name: str,
                 planes: list[str],
                 classes: list[str],
                 loss_func: Callable,
                 weight: float,
                 temperature: float = 0.):
        super().__init__()
        self.name = name
        self.planes = planes
        self.classes = classes
        self.loss_func = loss_func
        self.weight = weight
        self.temp = nn.Parameter(tensor(temperature))
        self.confusion = nn.ModuleDict()

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        # Describes how to arrange the elements of the batch to pass to the loss function. The output of this method is
        # what is fed to the loss function.
        raise NotImplementedError

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        raise NotImplementedError

    def loss(self,
             batch,  # batch of graphs
             stage: str,  # Network stage 'train' or 'test'
             confusion: bool = False):
        x, y = self.arrange(batch)
        w = self.weight * (-1 * self.temp).exp()
        loss = w * self.loss_func(x, y) + self.temp
        metrics = {}
        if stage:
            metrics = self.metrics(x, y, stage)
            metrics[f'loss_{self.name}/{stage}'] = loss
            if stage == 'train':
                metrics[f'temperature/{self.name}'] = self.temp
            if confusion:
                for cm in self.confusion.values():
                    cm.update(x, y)
        return loss, metrics

    def finalize(self, batch) -> None:
        # Function to apply to the decoder output after the loss is calculated
        return

    def draw_confusion_matrix(self, cm: tm.ConfusionMatrix) -> plt.Figure:
        '''Produce confusion matrix at end of epoch'''
        confusion = cm.compute().cpu()
        fig = plt.figure(figsize=[8, 6])
        sn.heatmap(confusion,
                   xticklabels=self.classes,
                   yticklabels=self.classes,
                   vmin=0, vmax=1,
                   annot=True)
        plt.ylim(0, len(self.classes))
        plt.xlabel('Assigned label')
        plt.ylabel('True label')
        return fig

    def on_epoch_end(self,
                     logger: 'pl.loggers.TensorBoardLogger',
                     stage: str,
                     epoch: int) -> None:

        if not logger: return
        for name, cm in self.confusion.items():
            logger.experiment.add_figure(
                f'{name}/{stage}',
                self.draw_confusion_matrix(cm),
                global_step=epoch)
            cm.reset()


class LogitBias(nn.Module):
    """Adds a learnable bias to each class logit"""

    def __init__(self, num_classes: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(num_classes))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.bias


class SemanticDecoder(DecoderBase):
    """NuGraph semantic decoder module.

    Convolve down to a single node score per semantic class for each 2D graph,
    node, and remove intermediate node stores from data object.
    """

    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('semantic',  # decoder name
                         planes,  # planes
                         semantic_classes,  # classes
                         RecallLoss(),  # loss function # BalancedFocalRecallLoss()
                         weight=2.)  # decoder weight

        # torchmetrics arguments
        metric_args = {
            'task': 'multiclass',
            'num_classes': len(semantic_classes),
            'ignore_index': -1
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_semantic_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_semantic_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Linear(node_features, len(semantic_classes))  # "Score" of each semantic class

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        # Note that each plane has its own semantic labeling. Usually the labeling is consistent across planes, but
        # there are cases where the same hit is labeled differently in different planes.
        return {'x_semantic': {p: self.net[p](x[p]) for p in self.planes}}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        # Concatenates each plane graph into a single big graph
        x = cat([batch[p].x_semantic for p in self.planes], dim=0)
        y = cat([batch[p].y_semantic for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'recall_semantic/{stage}': self.recall(x, y),
            f'precision_semantic/{stage}': self.precision(x, y)
        }

    def finalize(self, batch) -> None:
        for p in self.planes:
            batch[p].x_semantic = batch[p].x_semantic.softmax(dim=1)


class MichelBinaryDecoder(DecoderBase):
    """
    Per-node auxiliary classifier: for each hit, predict Michel (1) vs not-Michel (0).

    Uses BCEWithLogitsLoss (so targets are 0/1, predictions are raw logits).
    """

    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 michel_idx: int):
        super().__init__('michel_bin',
                         planes,
                         ['not_michel', 'michel'],
                         nn.BCEWithLogitsLoss(),
                         weight=1.0)

        self.michel_idx = michel_idx

        self.net = nn.ModuleDict({
            p: nn.Linear(node_features, 1) for p in planes
        })

        metric_args = {'task': 'binary'}
        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.ap = tm.AveragePrecision(**metric_args)
        self.confusion['recall_michel_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_michel_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return {'x_michel_bin': {p: self.net[p](x[p]).squeeze(-1) for p in self.planes}}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        preds, targets = [], []
        for p in self.planes:
            preds.append(batch[p].x_michel_bin)

            # 1 for Michel, 0 otherwise; ignore nodes with y=-1
            mask = (batch[p].y_semantic != -1)
            label = (batch[p].y_semantic == self.michel_idx).float()
            targets.append(label * mask.float())
        return cat(preds, dim=0), cat(targets, dim=0)

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        # x is logits; convert to probabilities for metrics
        probs = x.sigmoid()
        return {
            f'recall_michel_bin/{stage}': self.recall(probs, y),
            f'precision_michel_bin/{stage}': self.precision(probs, y),
            f'AP_michel/{stage}': self.ap(probs, y)
        }


class CountDecoder(DecoderBase):
    """
    Predict, per plane, the distribution (fractions) over semantic classes for each event.
    """
    def __init__(
        self,
        node_features: int,
        planes: list[str],
        semantic_classes: list[str],
        *,
        hidden: int = 64,
        shared_head: bool = True,
        plane_emb_dim: int = 16,
        temperature: float = 1.0,
        label_smoothing: float = 0.0,
        weight: float = 1e-3,
    ):
        super().__init__(
            'count',     # decoder name
            planes,            # planes
            semantic_classes,  # classes
            nn.MSELoss(),      # keep MSE to avoid breaking training loop
            weight=weight
        )

        self.semantic_classes = semantic_classes
        self.num_classes = len(semantic_classes)
        self.temperature = float(temperature)
        self.label_smoothing = float(label_smoothing)
        self.shared_head = bool(shared_head)
        self.planes = list(planes)
        self._plane_to_idx = {p: i for i, p in enumerate(self.planes)}

        self.mse = tm.MeanSquaredError()
        self.mae = tm.MeanAbsoluteError()

        # Poolers
        self.pool_sum  = SumAggregation()
        self.pool_mean = MeanAggregation()
        self.pool_attn = GlobalAttention(nn.Linear(node_features, 1)) # 1-layer gate

        # Feature dim after pooling: sum(H) + mean(H) + attn(H) + log|V|
        pooled_dim = 3 * node_features + 1

        if self.shared_head:
            self.plane_emb = nn.Embedding(len(self.planes), plane_emb_dim)
            in_dim = pooled_dim + plane_emb_dim
            self.head = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.LayerNorm(hidden),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden, self.num_classes),
            )
        else:
            in_dim = pooled_dim
            self.net = nn.ModuleDict({
                p: nn.Sequential(
                    nn.Linear(in_dim, hidden),
                    nn.LayerNorm(hidden),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(hidden, self.num_classes),
                ) for p in self.planes
            })

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        out = {}
        for p in self.planes:
            xs = self.pool_sum(x[p],  batch[p]) # (G, H)
            xm = self.pool_mean(x[p], batch[p]) # (G, H)
            xa = self.pool_attn(x[p], batch[p]) # (G, H)

            # log graph size feature to stabilize scale when node counts vary
            counts = torch.bincount(batch[p], minlength=xs.size(0)).float().unsqueeze(1) # (G,1)
            log_n  = torch.log(torch.clamp_min(counts, 1.0)) # (G,1)

            h = torch.cat([xs, xm, xa, log_n], dim=-1)# (G, pooled_dim)

            if self.shared_head:
                j = self._plane_to_idx[p]
                pe = self.plane_emb.weight[j].unsqueeze(0).expand(h.size(0), -1) # (G, E)
                logits = self.head(torch.cat([h, pe], dim=-1))  # (G, C)
            else:
                logits = self.net[p](h) # (G, C)

            out[p] = self.softmax(logits / self.temperature)

        return {'x_count': out}

    @torch.no_grad()
    def _counts_per_graph(self, labels: Tensor, g_idx: Tensor, num_graphs: int) -> Tensor:
        one_hot = F.one_hot(labels, num_classes=self.num_classes).float() # (N, C)
        sums = torch.zeros(num_graphs, self.num_classes, device=labels.device)
        sums.index_add_(0, g_idx, one_hot)  # (G, C)
        return sums

    def arrange(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        device = batch[self.planes[0]].x_count.device
        num_graphs = batch.num_graphs
        P = len(self.planes)
        C = self.num_classes

        x = torch.zeros(num_graphs, P, C, device=device)
        y = torch.zeros(num_graphs, P, C, device=device)

        for j, p in enumerate(self.planes):
            x[:, j, :] = batch[p].x_count

            labels = batch[p].y_semantic
            valid = (labels != -1)
            if valid.any():
                g_idx = batch[p].batch[valid]
                sums  = self._counts_per_graph(labels[valid], g_idx, num_graphs) # (G, C)

                # Convert to fractions
                totals = sums.sum(dim=1, keepdim=True) # (G,1)
                frac = torch.where(
                    totals > 0,
                    sums / torch.clamp_min(totals, 1.0),
                    torch.zeros_like(sums)
                )

                # optional label smoothing (only on rows with data)
                eps = self.label_smoothing
                if eps > 0:
                    frac = (1 - eps) * frac + eps / C

                y[:, j, :] = frac
            else:
                y[:, j, :] = 0.0

        # Flatten planes and classes
        x = x.flatten(start_dim=1)  # (G, P*C)
        y = y.flatten(start_dim=1)  # (G, P*C)
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'MSE_count/{stage}': self.mse(x, y),
            f'MAE_count/{stage}': self.mae(x, y)
        }


class FilterDecoder(DecoderBase):
    """NuGraph filter decoder module.

    Convolve down to a single node score, to identify and filter out
    graph nodes that are not part of the primary physics interaction
    """

    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 ):
        super().__init__('filter',  # decoder name
                         planes,  # planes
                         ('noise', 'signal'),  # classes
                         nn.BCELoss(),  # loss function
                         weight=1.)  # decoder weight

        # torchmetrics arguments
        metric_args = {
            'task': 'binary'
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_filter_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_filter_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(node_features, 1),
                nn.Sigmoid(),
            )

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return {'x_filter': {p: self.net[p](x[p]).squeeze(dim=-1) for p in self.planes}}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        # Concatenating all planes nodes into a single tensor
        x = cat([batch[p].x_filter for p in self.planes], dim=0)
        y = cat([(batch[p].y_semantic != -1).float() for p in self.planes], dim=0)  # How is batch[p].y_semantic stored?
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'recall_filter/{stage}': self.recall(x, y),
            f'precision_filter/{stage}': self.precision(x, y)
        }


class EventDecoder(DecoderBase):
    '''NuGraph event decoder module.

    Convolve graph node features down to a single classification score
    for the entire event
    '''

    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 event_classes: list[str]):
        super().__init__('event',
                         planes,
                         event_classes,
                         RecallLoss(),
                         weight=2.)

        # torchmetrics arguments
        metric_args = {
            'task': 'multiclass',
            'num_classes': len(event_classes)
        }

        self.recall = tm.Recall(**metric_args)
        self.precision = tm.Precision(**metric_args)
        self.confusion['recall_event_matrix'] = tm.ConfusionMatrix(
            normalize='true', **metric_args)
        self.confusion['precision_event_matrix'] = tm.ConfusionMatrix(
            normalize='pred', **metric_args)

        self.pool = nn.ModuleDict()
        for p in planes:
            self.pool[p] = SoftmaxAggregation(learn=True)
        self.net = nn.Sequential(
            nn.Linear(in_features=len(planes) * node_features,
                      out_features=len(event_classes)))

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        x = [pool(x[p], batch[p]) for p, pool in self.pool.items()]
        return {'x': {'evt': self.net(cat(x, dim=1))}}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        return batch['evt'].x, batch['evt'].y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'recall_event/{stage}': self.recall(x, y),
            f'precision_event/{stage}': self.precision(x, y)
        }

    def finalize(self, batch) -> None:
        batch['evt'].x = batch['evt'].x.softmax(dim=1)


class VertexDecoder(DecoderBase):
    """
    """

    def __init__(self,
                 node_features: int,
                 aggr: str,
                 lstm_features: int,
                 mlp_features: list[int],
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('vertex',
                         planes,
                         semantic_classes,
                         LogCoshLoss(),
                         weight=1.,
                         temperature=5.)

        # initialise aggregation function
        self.aggr = nn.ModuleDict()
        aggr_kwargs = {}
        in_features = node_features
        if aggr == 'lstm':
            aggr_kwargs = {
                'in_channels': node_features,
                'out_channels': lstm_features,
            }
            in_features = lstm_features
        for p in self.planes:
            self.aggr[p] = aggr_resolver(aggr, **(aggr_kwargs or {}))

        # initialise MLP
        net = []
        feats = [len(self.planes) * in_features] + mlp_features + [3]
        for f_in, f_out in zip(feats[:-1], feats[1:]):
            net.append(nn.Linear(in_features=f_in, out_features=f_out))
            net.append(nn.ReLU())
        del net[-1]  # remove last activation function
        self.net = nn.Sequential(*net)

    def forward(self, x: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        x = [net(x[p], index=batch[p]) for p, net in self.aggr.items()]
        x = cat(x, dim=1)
        return {'x_vtx': {'evt': self.net(x)}}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        x = batch['evt'].x_vtx
        y = batch['evt'].y_vtx
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        xyz = (x - y).abs().mean(dim=0)
        return {
            f'vertex-resolution-x/{stage}': xyz[0],
            f'vertex-resolution-y/{stage}': xyz[1],
            f'vertex-resolution-z/{stage}': xyz[2],
            f'vertex-resolution/{stage}': xyz.square().sum().sqrt()
        }


class InstanceDecoder(DecoderBase):
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 classes: list[str]):
        super().__init__('Instance',
                         planes,
                         event_classes,
                         ObjCondensationLoss(),
                         'multiclass',
                         confusion=False)

        num_features = len(classes) * node_features

        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(num_features, 1),
                nn.Sigmoid())

    def forward(self, x: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return {'x_instance': {p: self.net[p](x[p].flatten(start_dim=1)).squeeze(dim=-1) for p in self.net.keys()}}

    def arrange(self, batch: dict[str, Tensor]) -> tuple[Tensor, Tensor]:
        x = torch.cat([batch[p]['x_instance'] for p in self.planes], dim=0)
        y = torch.cat([batch[p]['y_instance'] for p in self.planes], dim=0)
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        metrics = {}
        predictions = self.predict(x)
        acc = self.acc_func(predictions, y)
        metrics[f'{self.name}_accuracy/{stage}'] = accuracy
        return metrics
