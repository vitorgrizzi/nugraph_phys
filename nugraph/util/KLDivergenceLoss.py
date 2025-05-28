import torch
import math
from torch import Tensor
import torch.nn.functional as F

class KLDivergenceLoss(torch.nn.Module):
    """
    KL divergence for two distributions p, q in [batch_size, n_classes].

    forward(pred, target) ->
        \sum_i p_i * log(p_i / q_i)  (averaged over the batch)

    Requirements:
    - pred.shape == target.shape == (N, C)
    - Both pred[i] and target[i] sum to 1 across classes, or at least
      are valid distributions.
    """

    def __init__(self, eps: float = 1e-8):
        """
        eps is a small constant added to clamp predictions & targets
        away from zero to avoid log(0).
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Check shapes
        assert pred.shape == target.shape, \
            f"pred shape {pred.shape} != target shape {target.shape}"
        assert pred.ndim == 2, "Expecting 2D input [batch, classes]"

        # Clamp both pred & target to avoid log(0)
        pred = torch.clamp(pred, min=self.eps)
        target = torch.clamp(target, min=self.eps)

        # KL(p||q) = sum( p_i log(p_i / q_i) ), averaged over batch
        kl_per_sample = torch.sum(target * (torch.log(target) - torch.log(pred)), dim=1)
        return kl_per_sample.mean()