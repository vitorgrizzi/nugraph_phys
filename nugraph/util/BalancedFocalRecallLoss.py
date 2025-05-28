import torch

CLASS_FREQ = [214454862, 34228660, 28152431, 3881675, 12530369]

class BalancedFocalRecallLoss(nn.Module):
    def __init__(self, class_freq=CLASS_FREQ, gamma=0.5, eps=1e-8):
        super().__init__()
        inv   = 1. / torch.clamp(torch.tensor(class_freq, dtype=torch.float32),
                                 min=1.0)
        alpha = inv / inv.sum()
        self.register_buffer("alpha", alpha)
        self.gamma = gamma
        self.eps   = eps

    def forward(self, logits, target):
        mask   = (target != -1)
        target = target[mask]
        logits = logits[mask]

        logp   = F.log_softmax(logits, dim=1)
        pt     = logp.exp().gather(1, target.unsqueeze(1)).squeeze(1)

        alpha_t = self.alpha[target]
        focal   = alpha_t * (1 - pt).pow(self.gamma) * (-torch.log(pt + self.eps))

        denom   = torch.zeros_like(self.alpha)
        denom.scatter_add_(0, target, torch.ones_like(target, dtype=torch.float32))
        denom   = torch.clamp(denom, min=1.0)

        loss_per = torch.zeros_like(self.alpha)
        loss_per.scatter_add_(0, target, focal)
        return (loss_per / denom).mean()