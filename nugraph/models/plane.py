from typing import Any, Callable

from torch import Tensor, cat
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from torch_geometric.nn import MessagePassing

class MessagePassing2D(MessagePassing):

    propagate_type = { 'x': Tensor }

    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 aggr: str = 'add'):
        super().__init__(node_dim=0, aggr=aggr)

        # Multiply by 2 because each of the two nodes exchanging the message has (in_feat+planar_feat) attributes
        feats = 2 * (in_features + planar_features)

        # psi
        self.edge_net = nn.Sequential(
            nn.Linear(feats, 1), # why not nn.Linear(feats, planar_features) and drop *x_j in the message function?
            nn.Sigmoid(),
        )
        # The output of edge_net can be seen as the weight of neighbor x_j to the embedding of x_i.

        ## Test 1: Each planar feature has its own sigmoid weight, multiply by x_h
        # self.edge_net1 = nn.Sequential(
        #     nn.Linear(feats, in_features + planar_features),
        #     nn.Sigmoid(),
        # )

        ## Test 2: Don't multiply the output of this network by x_j => Not good
        # self.edge_net2 = nn.Sequential(
        #     nn.Linear(feats, in_features + planar_features),
        #     nn.Tanh(),
        # )

        # phi
        self.node_net = nn.Sequential(
            nn.Linear(feats, planar_features),
            nn.Tanh(),
            nn.Linear(planar_features, planar_features),
            nn.Tanh(),
        )

    def forward(self, x: Tensor, edge_index: Tensor):
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor):
        # The message is just the scaled node features of the neighbor x_j since self.edge_net() returns a scalar c=[0,1]
        # due to the sigmoid activation.
        return self.edge_net(cat((x_i, x_j), dim=-1).detach()) * x_j # message has the same shape as x_j

    def update(self, aggr_out: Tensor, x: Tensor):
        return self.node_net(cat((x, aggr_out), dim=-1)) # 'aggr_out' has the same shape as 'x'


class PlaneNet(nn.Module):
    '''Module to convolve within each detector plane'''
    def __init__(self,
                 in_features: int,
                 planar_features: int,
                 planes: list[str],
                 aggr: str = 'add',
                 checkpoint: bool = True):
        super().__init__()

        self.checkpoint = checkpoint

        # Message passing between nodes of the same plane
        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = MessagePassing2D(in_features,
                                           planar_features,
                                           aggr)


    def ckpt(self, fn: Callable, *args) -> Any:
        if self.checkpoint and self.training:
            return checkpoint(fn, *args)
        else:
            # This is self.net[p](x[p], edge_index[p]). Thus, we are calling the forward method of self.net which is
            # an object from the MessagePassing2D class.
            return fn(*args)


    def forward(self, x: dict[str, Tensor], edge_index: dict[str, Tensor]) -> None:
        for p in self.net:
            # 'x' is {plane: (n_nodes, planar_feats+in_feats)}
            x[p] = self.ckpt(self.net[p], x[p], edge_index[p])
