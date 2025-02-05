from argparse import ArgumentParser
import warnings
import psutil

import torch
from torch import Tensor, cat, empty
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from pytorch_lightning import LightningModule
from torch_geometric.data import Batch, HeteroData
from torch_geometric.utils import unbatch
from .encoder import Encoder
from .plane import PlaneNet
from .nexus import NexusNet
from .decoders import SemanticDecoder, FilterDecoder, EventDecoder, VertexDecoder, MichelDecoder, CountDecoder
from ..util import MichelDistribution


class NuGraph2(LightningModule):
    """PyTorch Lightning module for model training.

    Wrap the base model in a LightningModule wrapper to handle training and
    inference, and compute training metrics."""

    def __init__(self,
                 in_features: int = 4,  # Are the 4 hit features: [WireID, PeakTime, Integral, RMS] in that order
                 planar_features: int = 128,
                 # These 4 features are encoded into a (planar_features) vector through encoder.py
                 nexus_features: int = 32,  # Number of features of the nexus nodes
                 vertex_aggr: str = 'lstm',
                 vertex_lstm_features: int = 64,
                 vertex_mlp_features: list[int] = [64],
                 planes: list[str] = ['u', 'v', 'y'],  # Name of each plane
                 semantic_classes: list[str] = ['MIP', 'HIP', 'shower', 'michel', 'diffuse'],
                 # Possible classes of each node for semantic head
                 event_classes: list[str] = ['numu', 'nue', 'nc'],  # Possible events for event head
                 num_iters: int = 5,  # Number of message passing rounds, i.e. layers in the GNN
                 event_head: bool = False,
                 # Decoder that classifies events according to event_classes. An event is all 3 graphs, one for each plane
                 semantic_head: bool = True,  # Base decoder that classifies hits according to semantic_classes
                 filter_head: bool = True,
                 # Decoder that filters hits not related to the primary interaction (detector noise and cosmic rays)
                 vertex_head: bool = False,
                 # Decoder to identify the 3D space point of the primary neutrino interaction
                 count_head: bool = False,
                 checkpoint: bool = False,
                 michelenergy_reg: bool = True,
                 reg_type: str = 'landau',
                 michel_reg_cte: float = 1e-2,
                 lr: float = 0.0005):
        super().__init__()

        warnings.filterwarnings("ignore", ".*NaN values found in confusion matrix.*")

        self.save_hyperparameters()

        self.planes = planes
        self.semantic_classes = semantic_classes
        self.event_classes = event_classes
        self.num_iters = num_iters
        self.lr = lr
        self.michelenergy_reg = michelenergy_reg
        self.reg_type = reg_type
        self.michel_id = 3
        self.michel_reg_cte = michel_reg_cte

        self.encoder = Encoder(in_features,
                               planar_features,
                               planes,
                               )

        self.plane_net = PlaneNet(in_features,
                                  planar_features,
                                  planes,
                                  checkpoint=checkpoint)

        self.nexus_net = NexusNet(planar_features,
                                  nexus_features,
                                  planes,
                                  checkpoint=checkpoint)

        self.decoders = []

        if event_head:
            self.event_decoder = EventDecoder(
                planar_features,
                planes,
                event_classes)
            self.decoders.append(self.event_decoder)

        if semantic_head:
            self.semantic_decoder = SemanticDecoder(
                planar_features,
                planes,
                semantic_classes)
            self.decoders.append(self.semantic_decoder)

        if filter_head:
            self.filter_decoder = FilterDecoder(
                planar_features,
                planes,
            )
            self.decoders.append(self.filter_decoder)

        if vertex_head:
            self.vertex_decoder = VertexDecoder(
                planar_features,
                vertex_aggr,
                vertex_lstm_features,
                vertex_mlp_features,
                planes,
                semantic_classes)
            self.decoders.append(self.vertex_decoder)

        if count_head:
            self.count_decoder = CountDecoder(
                planar_features,
                planes,
                semantic_classes  # assuming this is the column idx of michel in a tensor (nodes, in_features)
            )
            self.decoders.append(self.count_decoder)

        if len(self.decoders) == 0:
            raise Exception('At least one decoder head must be enabled!')

    def forward(self,
                x: dict[str, Tensor],  # {plane: node_feature_matrix}, tensor of shape (n_nodes, in_features)
                edge_index_plane: dict[str, Tensor],  # {plane: edge_index_matrix}, tensor of shape (2, n_edges)
                edge_index_nexus: dict[str, Tensor],
                nexus: Tensor,
                batch: dict[str, Tensor]  # {plane: batch_vector}, identifies the graph each node in the batch belong to
                ) -> dict[str, Tensor]:
        # Batch keeps track of which graph in the batch each node belongs to. For the semantic and filter
        # decoders 'batch' is not used because they operate on the node level, but for the event and vertex
        # decoders that operate on a graph level 'batch' is used to correctly assign nodes to their graphs.

        # Note that neutrino interaction event is described here as the union of three disjoint graphs, one for
        # each plane. The nodes represent detector hits (i.e. ionized electrons hiting that plane) and the node
        # features stores additional information of that hit such as its integral, width, hit time, and wire index.

        # Encodes the node features of the graph. The graph is represented by a dict whose keys are planes and values
        # are node features (integral, rms width, hit time, wire index). After encoding, instead of 4 node features,
        # there will be planar_features=128 features. Thus, the shape of 'x' changes from (n_nodes, in_features) to
        # (n_nodes, planar_features)
        m = self.encoder(x)  # Calling the forward method of Encoder class
        # 'm' is a dict {'plane_key': (n_nodes, planar_features)}

        for _ in range(self.num_iters):  # Number of layers in the GNN (i.e. MP rounds).
            # shortcut connect features
            for p in self.planes:
                # At each MP round expand the node embedding m[p] by adding the original "raw" node features x[p].
                # However, the shape of m[p] doesn't change at each GNN layer because the MLP that returns the updated
                # node embedding in the message passing scheme returns a tensor with the same shape as m[p] before
                # concatenate x[p]. x[p] is appended to m[p] to explicitly consider the original features in the
                # embedding of a node on each round, as well as multiple abstraction levels of the original features.
                # Thus, each MP round adds a new abstraction layer to the input features (even though this input
                # feature mixes with the m[p] data after going through the MLP).
                m[p] = torch.cat((m[p], x[p]), dim=-1)
                # m[p] has shape (n_nodes, planar_feats + in_feats)

            # These two lines perform the message passing and automatically update m[p] to (n_nodes, planar_features).
            self.plane_net(m, edge_index_plane)
            self.nexus_net(m, edge_index_nexus, nexus)

        ret = {}
        for decoder in self.decoders:
            ret.update(decoder(m, batch))  # semantic and filter decoders don't use batch in their forward method.

        return ret

    def step(self, data: HeteroData | Batch,
             # HeteroData because nodes of different planes are considered different types
             stage: str = None,
             confusion: bool = False):

        # if it's a single data instance, convert to batch manually
        if isinstance(data, Batch):
            batch = data
        else:
            batch = Batch.from_data_list([data])  # construct

        # Unpack tensors to call forward method. 'x' is the output of the forward method, and it is a dict of dicts
        # {{decoder1: {plane: pred}}, {decoder2: {plane: pred}}} where 'pred' is a tensor (n_nodes, n_classes_attr)
        # returned by the forward method of the particular decoder
        x = self(batch.collect('x'),
                 # returns the node feature matrix 'x' of each plane {'u': u_node_feat_matrix, ... }
                 {p: batch[p, 'plane', p].edge_index for p in self.planes},
                 {p: batch[p, 'nexus', 'sp'].edge_index for p in self.planes},
                 torch.empty(batch['sp'].num_nodes, 0),
                 {p: batch[p].batch for p in self.planes})
        # All args besides the tensor torch.empty(batch['sp'].num_nodes, 0) are dicts {plane: tensor}.

        # Update the data input object with the decoder output tensors
        if isinstance(data, Batch):
            dlist = [HeteroData() for _ in range(data.num_graphs)]  # create an HeteroData object for each graph
            for attr, planes in x.items():  # x = {decoder_name: {u: tensor, v: tensor, y: tensor}, ...} is the output of forward()
                for p, t in planes.items():  # iterate to get a plane and a tensor
                    if t.size(0) == data[p].num_nodes:
                        tlist = unbatch(t, data[p].batch)
                    elif t.size(0) == data.num_graphs:  # data.num_graph is same as data.batch_size
                        tlist = unbatch(t, torch.arange(data.num_graphs))
                    else:
                        raise Exception(f'don\'t know how to unbatch attribute {attr}')

                    # At the end of the for loop below, each HeteroData in the batch will store the output of each
                    # decoder for each plane ['u', 'v', 'y'] as:
                    # HeteroData( u={ decoder1: decoder1_out_tensor, ..., decoderN: decoderN_out_tensor }
                    #             v={ decoder1: decoder1_out_tensor, ..., decoderN: decoderN_out_tensor }
                    #             y={ decoder1: decoder1_out_tensor, ..., decoderN: decoderN_out_tensor } )
                    for it_d, it_t in zip(dlist, tlist):
                        # For each HeteroData object in dlist, do HeteroDataObj[plane][decoder_name] = decoder_out_tensor,
                        # which creates an entry 'plane={decoder_name: decoder_out_tensor}' in the HeteroData object.
                        it_d[p][attr] = it_t  # same as it_d[p].attr = it_t
                    # The decoder output is {attr: {plane: tensor}} but we change its format to {plane: {attr: tensor}}

            tmp = Batch.from_data_list(dlist)
            data.update(tmp)
            for attr, planes in x.items():
                for p in planes:
                    data._slice_dict[p][attr] = tmp._slice_dict[p][attr]
                    data._inc_dict[p][attr] = tmp._inc_dict[p][attr]
            # 'data' is updated instead of using 'tmp' directly because 'tmp' only stores node features whereas data
            # also has information on edges.

        else:
            for key, value in x.items():
                data.set_value_dict(key, value)

        total_loss = 0.
        total_metrics = {}
        for decoder in self.decoders:
            # 'data' is a batch of HeteroData objects. Each HeteroData has the same format:
            # HeteroData( u={ decoder1: decoder1_out_tensor, ..., decoderN: decoderN_out_tensor }
            #             v={ decoder1: decoder1_out_tensor, ..., decoderN: decoderN_out_tensor }
            #             y={ decoder1: decoder1_out_tensor, ..., decoderN: decoderN_out_tensor } )
            loss, metrics = decoder.loss(data, stage, confusion)
            total_loss += loss
            total_metrics.update(metrics)
            decoder.finalize(data)

        # Michel Energy Regularization
        # Get the integral of all michel hits within an event and sum them. Then, use this sum to predict the deposited
        # energy according to a linear relation that I've derived from the h5 dataset. Note that we don't even need to
        # use `edep`, we can use the regularization with the integral directly since they are related by a constant.
        if self.michelenergy_reg:
            michel_reg_loss = 0.0
            edep_michel = 0.0

            # Hyperparams to tune
            edep_lim_high = 160
            edep_lim_low = 1
            pdf_amp = 10

            for p in self.planes:
                # Extract predicted labels across all graphs in the batch
                y_pred = torch.argmax(batch[p].x_semantic, dim=1)

                # Extract integral feature for nodes classified as Michel electrons
                sumintegral_michel = torch.sum(batch[p].x_raw[y_pred == self.michel_id, 2])  # integral is feature index 2

                # Compute deposited energy for the batch (normalized per graph)
                edep_michel += (sumintegral_michel * 0.00580717 / batch.num_graphs)

                if edep_michel > 0:
                    # Adding a penalty to the loss based on the predicted deposited energy and its expected value
                    if self.reg_type == 'cutoff':  # hard cutoff for very high/low deposited energies
                        if edep_michel > edep_lim_high:
                            michel_reg_loss += self.michel_reg_cte * (edep_michel - edep_lim_high) / 15
                        if edep_michel < edep_lim_low:
                            michel_reg_loss += self.michel_reg_cte * (edep_michel - edep_lim_low) / 10

                    elif self.reg_type == 'landau' and edep_michel > 8.5:  # single peak distribution
                        pdf_value = MichelDistribution.get_pdf_value(edep_michel, distribution='landau')
                        michel_reg_loss += self.michel_reg_cte * (1 - pdf_value) * pdf_amp

                    elif self.reg_type == 'data':  # purely from data, double peaked distribution
                        pdf_value = MichelDistribution.get_pdf_value(edep_michel, distribution='data')
                        michel_reg_loss += self.michel_reg_cte * (1 - pdf_value) * pdf_amp

                # # Extracting the true deposited energies
                # true_mich_idxs = torch.nonzero(graph[p].y_semantic == self.michel_id)
                # int += torch.sum(graph[p].x_raw[true_mich_idxs, 2])
                # if int != 0: print(f'Edep: {int * 0.00580717}')

            total_loss += michel_reg_loss

        return total_loss, total_metrics

    def on_train_start(self):
        hpmetrics = {'max_lr': self.hparams.lr}
        self.logger.log_hyperparams(self.hparams, metrics=hpmetrics)
        self.max_mem_cpu = 0.
        self.max_mem_gpu = 0.

        scalars = {
            'loss': {'loss': ['Multiline', ['loss/train', 'loss/val']]},
            'acc': {}
        }
        for c in self.semantic_classes:
            scalars['acc'][c] = ['Multiline', [
                f'semantic_accuracy_class_train/{c}',
                f'semantic_accuracy_class_val/{c}'
            ]]
        self.logger.experiment.add_custom_scalars(scalars)

    def training_step(self,
                      batch,
                      batch_idx: int) -> float:
        loss, metrics = self.step(batch, 'train')
        self.log('loss/train', loss, batch_size=batch.num_graphs, prog_bar=True)
        self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log_memory(batch, 'train')
        return loss

    def validation_step(self,
                        batch,
                        batch_idx: int) -> None:
        loss, metrics = self.step(batch, 'val', True)
        self.log('loss/val', loss, batch_size=batch.num_graphs)
        self.log_dict(metrics, batch_size=batch.num_graphs)

    def on_validation_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'val', epoch)

    def test_step(self,
                  batch,
                  batch_idx: int = 0) -> None:
        loss, metrics = self.step(batch, 'test', True)
        self.log('loss/test', loss, batch_size=batch.num_graphs)
        self.log_dict(metrics, batch_size=batch.num_graphs)
        self.log_memory(batch, 'test')

    def on_test_epoch_end(self) -> None:
        epoch = self.trainer.current_epoch + 1
        for decoder in self.decoders:
            decoder.on_epoch_end(self.logger, 'test', epoch)

    def predict_step(self,
                     batch: Batch,
                     batch_idx: int = 0) -> Batch:
        self.step(batch)
        return batch

    def configure_optimizers(self) -> tuple:
        optimizer = AdamW(self.parameters(),
                          lr=self.lr)
        onecycle = OneCycleLR(
            optimizer,
            max_lr=self.lr,
            total_steps=self.trainer.estimated_stepping_batches)
        return [optimizer], {'scheduler': onecycle, 'interval': 'step'}

    def log_memory(self, batch: Batch, stage: str) -> None:
        # log CPU memory
        if not hasattr(self, 'max_mem_cpu'):
            self.max_mem_cpu = 0.
        cpu_mem = psutil.Process().memory_info().rss / float(1073741824)
        self.max_mem_cpu = max(self.max_mem_cpu, cpu_mem)
        self.log(f'memory_cpu/{stage}', self.max_mem_cpu,
                 batch_size=batch.num_graphs, reduce_fx=torch.max)

        # log GPU memory
        if not hasattr(self, 'max_mem_gpu'):
            self.max_mem_gpu = 0.
        if self.device != torch.device('cpu'):
            gpu_mem = torch.cuda.memory_reserved(self.device)
            gpu_mem = float(gpu_mem) / float(1073741824)
            self.max_mem_gpu = max(self.max_mem_gpu, gpu_mem)
            self.log(f'memory_gpu/{stage}', self.max_mem_gpu,
                     batch_size=batch.num_graphs, reduce_fx=torch.max)

    @staticmethod
    def add_model_args(parser: ArgumentParser) -> ArgumentParser:
        '''Add argparse argpuments for model structure'''
        model = parser.add_argument_group('model', 'NuGraph2 model configuration')
        model.add_argument('--planar-feats', type=int, default=128,
                           help='Hidden dimensionality of planar convolutions')
        model.add_argument('--nexus-feats', type=int, default=32,
                           help='Hidden dimensionality of nexus convolutions')
        model.add_argument('--vertex-aggr', type=str, default='lstm',
                           help='Aggregation function for vertex decoder')
        model.add_argument('--vertex-lstm-feats', type=int, default=32,
                           help='Hidden dimensionality of vertex LSTM aggregation')
        model.add_argument('--vertex-mlp-feats', type=int, nargs='*', default=[32],
                           help='Hidden dimensionality of vertex decoder')
        model.add_argument('--event', action='store_true', default=False,
                           help='Enable event classification head')
        model.add_argument('--semantic', action='store_true', default=False,
                           help='Enable semantic segmentation head')
        model.add_argument('--filter', action='store_true', default=False,
                           help='Enable background filter head')
        model.add_argument('--vertex', action='store_true', default=False,
                           help='Enable vertex regression head')
        return parser

    @staticmethod
    def add_train_args(parser: ArgumentParser) -> ArgumentParser:
        train = parser.add_argument_group('train', 'NuGraph2 training configuration')
        train.add_argument('--no-checkpointing', action='store_true', default=False,
                           help='Disable checkpointing during training')
        train.add_argument('--epochs', type=int, default=80,
                           help='Maximum number of epochs to train for')
        train.add_argument('--learning-rate', type=float, default=0.001,
                           help='Max learning rate during training')
        train.add_argument('--clip-gradients', type=float, default=None,
                           help='Maximum value to clip gradient norm')
        train.add_argument('--gamma', type=float, default=2,
                           help='Focal loss gamma parameter')
        return parser
