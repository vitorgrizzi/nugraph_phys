import torch
import math
from torch import Tensor
import torch.nn.functional as F

class CrossEntropyDistributionLoss(torch.nn.Module):
    """
    Cross-entropy for two distributions p, q in [batch_size, n_classes].

    forward(pred, target) ->
        -\sum_i [ p_i * log(q_i) ]  (averaged over the batch)

    Requirements:
    - pred.shape == target.shape == (N, C)
    - Both pred[i] and target[i] sum to 1 across classes, or at least
      represent valid distributions.
    """

    def __init__(self, eps: float = 1e-8):
        """
        eps is a small constant added to clamp predictions
        away from zero to avoid log(0).
        """
        super().__init__()
        self.eps = eps

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        # Check shapes
        assert pred.shape == target.shape, \
            f"pred shape {pred.shape} != target shape {target.shape}"
        assert pred.ndim == 2, "Expecting 2D input [batch, classes]"

        # Clamp pred to avoid log(0); typically you don't clamp target
        # except to ensure no NaNs, but it's okay to clamp minimally if needed
        pred = torch.clamp(pred, min=self.eps)

        # Cross-entropy:  H(p, q) = -\sum_i p_i log(q_i)
        ce_per_sample = -torch.sum(target * torch.log(pred), dim=1)
        return ce_per_sample.mean()