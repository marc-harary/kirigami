import string
import torch

GRID = torch.linspace(0, 1, 100)
MIN_DIST = 4
BASE_PRIMES = torch.Tensor([2, 3, 5, 7])
PAIRS = {14, 15, 35}
PSEUDO_LEFT = "({[<" + string.ascii_uppercase
PSEUDO_RIGHT = ")}]>" + string.ascii_lowercase
BASE_DICT = dict(
    A=torch.Tensor([1, 0, 0, 0]),
    C=torch.Tensor([0, 1, 0, 0]),
    G=torch.Tensor([0, 0, 1, 0]),
    U=torch.Tensor([0, 0, 0, 1]),
    N=torch.Tensor([0.25, 0.25, 0.25, 0.25]),
    D=torch.Tensor([1 / 3, 0, 1 / 3, 1 / 3]),
    W=torch.Tensor([0.5, 0, 0, 0.5]),
    V=torch.Tensor([1 / 3, 1 / 3, 1 / 3, 0]),
    K=torch.Tensor([0, 0, 0.5, 0.5]),
    R=torch.Tensor([0.5, 0, 0.5, 0]),
    M=torch.Tensor([0.5, 0.5, 0, 0]),
    S=torch.Tensor([0, 0.5, 0.5, 0]),
    Y=torch.Tensor([0, 0.5, 0, 0.5]),
)
