import torch

__all__ = ['ActDropNorm']

class ActDropNorm(torch.nn.Module):
    '''Performs activation, dropout, and batch normalization for resnet blocks'''
    def __init__(self, p: float, activation='ReLU', num_channels=8):
        super(ActDropNorm, self).__init__()
        activation_class = getattr(nn, activation)
        self.act = activation_class()
        self.drop = torch.nn.Dropout(p=p)
        self.norm = torch.nn.BatchNorm2d(num_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        out = input
        out = self.act(out)
        out = self.drop(out)
        out = self.norm(out)
        return out
