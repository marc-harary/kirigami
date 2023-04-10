import torch
import string

MIN_DIST = 4
BASE_PRIMES = torch.Tensor([2, 3, 5, 7])
PAIRS = {14, 15, 35}
PSEUDO_LEFT = "({[<" + string.ascii_uppercase
PSEUDO_RIGHT = ")}]>" + string.ascii_lowercase
BASE_DICT = dict(
    A=torch.tensor([1, 0, 0, 0]),
    C=torch.tensor([0, 1, 0, 0]),
    G=torch.tensor([0, 0, 1, 0]),
    U=torch.tensor([0, 0, 0, 1]),
    N=torch.tensor([0.25, 0.25, 0.25, 0.25]),
    D=torch.tensor([1 / 3, 0, 1 / 3, 1 / 3]),
    W=torch.tensor([0.5, 0, 0, 0.5]),
    V=torch.tensor([1 / 3, 1 / 3, 1 / 3, 0]),
    K=torch.tensor([0, 0, 0.5, 0.5]),
    R=torch.tensor([0.5, 0, 0.5, 0]),
    M=torch.tensor([0.5, 0.5, 0, 0]),
    S=torch.tensor([0, 0.5, 0.5, 0]),
    Y=torch.tensor([0, 0.5, 0, 0.5]),
)
