import torch
from torch import Tensor

# Should I make intervals of class, e.g:
# class 1: 0   michel
# class 2: 1-5 michel
# ... and so on?

# Should I use the percentager of michel hits in the graph, or do it by plane and then taking an average? Using
# percentage could cause problems if there is a large imbalance in the number of hits across different planes.
#

# This is a graph/plane-level prediction (total number of michel electrons)
class MichelLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x: Tensor, y: Tensor, michel_idx: int=2):
         return torch.mean(y - x)

