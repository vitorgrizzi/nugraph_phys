from typing import Any, Callable

from abc import ABC

import torch
from torch import Tensor, tensor, cat
import torch.nn as nn
from torch_geometric.nn.aggr import SoftmaxAggregation, LSTMAggregation, SumAggregation, MeanAggregation, MaxAggregation, MinAggregation
from torch_geometric.nn.resolver import aggregation_resolver as aggr_resolver

import torchmetrics as tm

import matplotlib.pyplot as plt
import seaborn as sn

from ..util import RecallLoss, LogCoshLoss, ObjCondensationLoss, MichelLoss


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
             batch, # batch of graphs
             stage: str, # Network stage 'train' or 'test'
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
        fig = plt.figure(figsize=[8,6])
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


class SemanticDecoder(DecoderBase):
    """NuGraph semantic decoder module.

    Convolve down to a single node score per semantic class for each 2D graph,
    node, and remove intermediate node stores from data object.
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]):
        super().__init__('semantic', # decoder name
                         planes, # planes
                         semantic_classes, # classes
                         RecallLoss(), # loss function
                         weight=2.) # decoder weight

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
            self.net[p] = nn.Linear(node_features, len(semantic_classes)) # "Score" of each semantic class

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        # Note that each plane has its own semantic labeling. Usually the labeling is consistent across planes, but
        # there are cases where the same hit is labeled differently in different planes.
        return { 'x_semantic': { p: self.net[p](x[p]) for p in self.planes } }

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


class MichelDecoder(DecoderBase):
    """
    Asserts if the energy distribution (integral label) of michel electron hits follows the expected physics?

    As a low-hanging fruit, I will first compare the number of hits labeled as michel electron instead of the number of
    michel electrons itself (a single michel electron produces many hits). Thus, in this case the loss function will
    compare predicted percentage of michel electrons hits with the true number of michel electrons hits.
    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 michel_id: int
                 ):

        super().__init__('michel', # decoder name
                         planes, # planes
                         ('michel', 'not_michel'), # classes
                         nn.L1Loss(reduction='mean'), # loss function
                         weight=2.) # decoder weight


        self.michel_id = michel_id
        self.mse = tm.MeanSquaredError()
        self.mae = tm.MeanAbsoluteError()

        self.pool = SumAggregation() # don't need to use a pool for each plane because its a simply sum aggr
        self.net = nn.ModuleDict() # self.net here is a readout function
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(node_features, 1),
                nn.Sigmoid()
            )
        # Deeper the MLP in the decoder is, more it can learn through adjusting its own parameters instead of adjusting
        # the parameters before the decoder heads. Thus, a deeper decoder MLP means less "interaction" between decoders.

    # Outputs the percentage of michel electrons hits on each plane
    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        return { 'x_michel': { p: self.net[p]( self.pool(x[p], batch[p]) ) for p in self.planes } }

    # I think I'll have to open that batch and extract them. In this case 'x' would be shape (n_graphs, 3) and 'y'
    # (n_graphs, 1) where 'n_graphs' are the number of graphs in the batch.
    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        # x = cat([batch[p].x_michel for p in self.planes], dim=0) # Each event graph yield a shape (3,) vector, one for each plane

        ## Must unbatch the 'y' tensor as {'u': tensor_list, 'v': tensor_list, 'y': tensor_list} using batch._slice_dict
        ## to obtain the total number of michel hits of each graph.
        x = torch.empty(batch.num_graphs, len(self.planes))
        y = torch.empty(batch.num_graphs, len(self.planes))
        for i, graph in enumerate(batch.to_data_list()):
            planes_semantic = graph.collect('y_semantic')
            planes_count = graph.collect('x_michel')
            for j, p in enumerate(self.planes):
                y[i,j] = torch.count_nonzero(planes_semantic[p] == self.michel_id) / planes_semantic[p].size(0)
                x[i,j] = planes_count[p]
        y = y.flatten(-2,-1)
        x = x.flatten(-2,-1)

        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        return {
            f'MSE_michel/{stage}': self.mse(x, y),
            f'MAE_michel/{stage}': self.mae(x, y)
        }

class CountDecoder(DecoderBase):
    """
    Train the decoder to predict the percentage of each labeled class

    """
    def __init__(self,
                 node_features: int,
                 planes: list[str],
                 semantic_classes: list[str]
                 ):

        super().__init__('count', # decoder name
                         planes, # planes
                         semantic_classes, # classes
                         nn.L1Loss(reduction='mean'), # loss function
                         weight=1.) # decoder weight

        self.semantic_classes = semantic_classes
        self.mse = tm.MeanSquaredError()
        self.mae = tm.MeanAbsoluteError()

        # don't need to use a different aggregation for each plane because its a simply sum pooling, nothing to "learn"
        self.pool_sum = SumAggregation()
        self.pool_mean = MeanAggregation()
        self.pool_min = MinAggregation()
        self.pool_max = MaxAggregation()
        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                # nn.Linear(4*node_features, node_features),
                nn.Linear(4*node_features, len(semantic_classes)),
                nn.Softmax() # softmax to predict directly the percentages
            )
        # Deeper the MLP in the decoder is, more it can learn through adjusting its own parameters instead of adjusting
        # the parameters before the decoder heads. Thus, a deeper decoder MLP means less "interaction" between decoders.

    def forward(self, x: dict[str, Tensor],
                batch: dict[str, Tensor]) -> dict[str, dict[str, Tensor]]:
        out = {}

        ## Concatenate different pools
        for p in self.planes:
            # The pools return a matrix (n_graphs_in_batch, node_features)
            inp_sum = self.pool_sum(x[p], batch[p])
            inp_mean = self.pool_mean(x[p], batch[p])
            inp_min = self.pool_min(x[p], batch[p])
            inp_max = self.pool_max(x[p], batch[p])
            # sizes = torch.unique(batch[p], return_counts=True)[1].view(-1,1)
            inp = cat((inp_sum, inp_mean, inp_min, inp_max), dim=-1)
            out.update({p: self.net[p](inp)})

        return {'x_count': out}
        # return { 'x_count': { p: self.net[p]( self.pool_sum(x[p], batch[p]) ) for p in self.planes } }

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        x = torch.zeros(batch.num_graphs, len(self.planes), len(self.semantic_classes))
        y = torch.zeros(batch.num_graphs, len(self.planes), len(self.semantic_classes))
        for i, graph in enumerate(batch.to_data_list()):
            planes_semantic = graph.collect('y_semantic')
            planes_count = graph.collect('x_count')
            for j, p in enumerate(self.planes):
                hits_id = planes_semantic[p][planes_semantic[p] != -1]
                keys, counts = torch.unique(hits_id, return_counts=True)
                counts_perc = counts.float() / counts.sum()
                for idx, k in enumerate(keys):
                    y[i,j,k] = counts_perc[idx]
                x[i,j] = planes_count[p]

                print(y[i,j], x[i,j])
                print(y[i,j] - x[i,j])

        y = y.flatten(start_dim=1)
        x = x.flatten(start_dim=1)

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
        super().__init__('filter', # decoder name
                         planes, # planes
                         ('noise', 'signal'), # classes
                         nn.BCELoss(), # loss function
                         weight=2.) # decoder weight

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
        return { 'x_filter': { p: self.net[p](x[p]).squeeze(dim=-1) for p in self.planes }}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        # Concatenating all planes nodes into a single tensor
        x = cat([batch[p].x_filter for p in self.planes], dim=0)
        y = cat([(batch[p].y_semantic!=-1).float() for p in self.planes], dim=0) # How is batch[p].y_semantic stored?
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
        x = [ pool(x[p], batch[p]) for p, pool in self.pool.items() ]
        return { 'x': { 'evt': self.net(cat(x, dim=1)) }}

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
        feats = [ len(self.planes) * in_features ] + mlp_features + [ 3 ]
        for f_in, f_out in zip(feats[:-1], feats[1:]):
            net.append(nn.Linear(in_features=f_in, out_features=f_out))
            net.append(nn.ReLU())
        del net[-1] # remove last activation function
        self.net = nn.Sequential(*net)

    def forward(self, x: dict[str, Tensor], batch: dict[str, Tensor]) -> dict[str,dict[str, Tensor]]:
        x = [ net(x[p], index=batch[p]) for p, net in self.aggr.items() ]
        x = cat(x, dim=1)
        return { 'x_vtx': { 'evt': self.net(x) }}

    def arrange(self, batch) -> tuple[Tensor, Tensor]:
        x = batch['evt'].x_vtx
        y = batch['evt'].y_vtx
        return x, y

    def metrics(self, x: Tensor, y: Tensor, stage: str) -> dict[str, Any]:
        xyz = (x-y).abs().mean(dim=0)
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
