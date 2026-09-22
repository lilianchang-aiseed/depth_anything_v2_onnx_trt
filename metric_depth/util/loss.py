import torch
from torch import nn


class SiLogLoss(nn.Module):
    def __init__(self, lambd=0.5):
        super().__init__()
        self.lambd = lambd

    def forward(self, pred, target, valid_mask):
        valid_mask = valid_mask.detach()
        diff_log = torch.log(target[valid_mask]) - torch.log(pred[valid_mask])
        loss = torch.sqrt(torch.pow(diff_log, 2).mean() -
                          self.lambd * torch.pow(diff_log.mean(), 2))

        return loss


class RangeWeightedSiLogLoss(nn.Module):
    """Global SiLog plus independently weighted SiLog depth bands."""

    RANGES = (
        ("0.2_2m", 0.2, 2.0),
        ("2_5m", 2.0, 5.0),
        ("5_10m", 5.0, 10.0),
        ("10_20m", 10.0, 20.000001),
    )

    def __init__(self, lambd=0.5, weights=(0.4, 0.3, 0.2, 0.1),
                 min_pixels=10):
        super().__init__()
        if len(weights) != len(self.RANGES):
            raise ValueError(f"expected {len(self.RANGES)} range weights")
        self.base_loss = SiLogLoss(lambd=lambd)
        self.weights = tuple(float(weight) for weight in weights)
        self.min_pixels = int(min_pixels)

    def forward(self, pred, target, valid_mask, return_components=False):
        global_loss = self.base_loss(pred, target, valid_mask)
        total = global_loss
        components = {"global": global_loss}

        for (name, low, high), weight in zip(self.RANGES, self.weights):
            range_mask = (valid_mask & (target >= low) & (target < high))
            if int(range_mask.sum()) >= self.min_pixels:
                range_loss = self.base_loss(pred, target, range_mask)
            else:
                range_loss = pred.sum() * 0.0
            components[name] = range_loss
            total = total + weight * range_loss

        components["combined"] = total
        return (total, components) if return_components else total
