# as described in https://arxiv.org/abs/2106.14917

import torch
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.functional import recall

# Recall tells the percentage of True labels that were correctly classified as True, i.e. recall = TP/(TP+FN).
class RecallLoss(torch.nn.Module):
    def __init__(self, ignore_index: int=-1):
        super().__init__()
        self.ignore_index = ignore_index

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        weight = 1 - recall(input, target, 'multiclass',
                            num_classes=input.size(1),
                            average='none', # average='none' is to compute the recall for each class individually
                            ignore_index=self.ignore_index)

        ce = F.cross_entropy(input, target, reduction='none',
                             ignore_index=self.ignore_index)

        # Penalizes the cross_entropy function according to the recall of each class. If the recall of one class is
        # really bad, its weight gonna be higher since 'weight = 1 - recall'. The effect of this is to balance the
        # recall errors between classes. The network will have a stronger incentive to perform parameters updates that
        # improve the worst performing class recall (even at the expense of decreasing the others classes recall).
        loss = weight[target] * ce

        return loss.mean()