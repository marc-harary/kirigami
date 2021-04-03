import torch
import torch.nn

__all__ ["InverseLoss"]

class InverseLoss(nn.Module):
    def __init__(self, A: float =  20., epsilon: float = 1e-1):
        super().__init__()
        self.A = A
        self.epsilon = epsilon

    def forward(self, pred: torch.Tensor, grnd: torch.Tensor) -> torch.Tensor:
        # which loss function to use now?
        pred_inv = self.A / (pred + self.epsilon)
        grnd_inv = self.A / (grnd + self.epsilon)
        return ((pred_inv - grnd_inv)**2).sum()
