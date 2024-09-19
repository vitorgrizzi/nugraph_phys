from torch import Tensor
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self,
                 in_features: int, # number of original node features (4)
                 node_features: int, # same as planar_features, is the number of embbeded node features
                 planes: list[str]): # number of planes (3)
        super().__init__()

        # This ModuleDict stores an encoder for each plane. The keys for the ModuleDict below are the planes [u, v, y].
        # Each key stores a nn.Sequential, which is the encoder itself.
        self.net = nn.ModuleDict()
        for p in planes:
            self.net[p] = nn.Sequential(
                nn.Linear(in_features, node_features),
                nn.Tanh(),
            )

    def forward(self, x: dict[str, Tensor]) -> dict[str, Tensor]:
        # Returns a dict whose keys represent planes [u, v, y], and values are NNs that take the hit attributes
        # stored in each node and return an embedding of this node. Each plane has its own encoder (i.e. its own
        # NN with different weights).
        return { p: net(x[p]) for p, net in self.net.items() }
        # The tensor associated with each plane key has shape (n_nodes, node_features)