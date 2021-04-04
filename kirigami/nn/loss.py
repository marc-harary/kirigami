import torch
import torch.nn as nn

__all__ = ["InverseLoss"]

# MAX_DIST = 22
# MIN_DIST = 8
# INTERVAL_DIST = 0.1
# Embedding = torch.eye(((MAX_DIST
# for i in range(8, 22.1, .1):
#     Em

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


class LossEmbedding(nn.Module):
    def __init__(self,
                 max_dist: float = 22.,
                 min_dist: float = 8.,
                 step_dist: float = 0.1):
        super().__init__()
        self.max_dist = max_dist
        self.min_dist = min_dist
        self.step_dist = step_dist 
        self.embedding_dict = torch.eye((self.max_dist - self.min_dist) / self.step_dist)
    
    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(ipt)
        out.unsqueeze_(-1)
        out = out.expand(-1)
        out = out.expand(self.embedding_dict.shape[0])
        for i in range(ipt.shape(-2)):
            for j in range(ipt.shape(-1)):
                dist = ipt[:, i, j]
                idx = (dist - self.min_dist) / self.step_dist
                ipt[:, i, j] = self.embedding_dict[idx]

        

# class OneHotLoss(nn.Module):
#     def __init__(self):
#         super().__init__()
# 
#     def forward(self, pred: torch.Tensor, grnd: torch.Tensor) -> torch.Tensor:
        
