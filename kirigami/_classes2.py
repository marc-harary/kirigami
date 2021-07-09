from typing import OrderedDict, Tuple
from dataclasses import dataclass


PdbStr = NewType("PdbStr", str)
StStr = NewType("StStr", str)
BpseqStr = NewType("BpseqStr", str)


class Distance:

    @dataclass
    class Pair:
        PP: float
        O5O5: float 
        C5C5: float
        C4C4: float
        C3C3: float
        C2C2: float
        C1C1: float
        O4O4: float
        O3O3: float
        NN: float

    def __init__(self, pairs: OrderedDict[Distance.Pair]) -> None:
        self._nucleotides = int(len(pairs)**.5)
        self.pairs = pairs

    @property 
    def nucleotides(self) -> int:
        return self._nucleotides
    
    def __getitem__(self, idx: Tuple[int,int]) -> Distance.Pair:
        return self.pairs[idx]

    def __len__(self) -> int:
        return len(self.pairs)

    def __iter__(self) -> Distance:
        self.idx = 0
        return self

    def __next__(self) -> Distance.Pair:
        if self.idx == self.nucleotides:
            raise StopIteration
        ii = self.idx // self.nucleotides
        jj = self.idx % self.nucleotides
        self.idx += 1
        return self[ii,jj]

    # def _to_float(self,
    #               dim: int = 3,
    #               pad_length: int = 0,
    #               dtype: torch.dtype = torch.float,
    #               device: torch.device = torch.device("cpu")) -> torch.Tensor:
    #     out = torch.zeros((10, L, L))
    #     for (i, j), dist in self.items():
    #         out[:,i,j] = torch.tensor([getattr(dist, field) for field in fields])
    #     while out.dim() < dim:
    #         out.unsqueeze_(0)
    #     return out

    @classmethod
    def from_pdb(cls, pdb: PdbStr) -> Distance:
        lines = pdb.copy().splitlines()
        del lines[0] # drop header
        c = -2*len(lines) + 2
        dis = 1 - 4*c
        sqrt_val = dis**.5
        L = int((1+sqrt_val) / 2)
        out_unsort = {}
        for line in lines:
            words = line.split()
            i = int(words[2]) - 1
            j = int(words[5]) - 1
            dist_list = list(map(float, words[-10:]))
            dist = Pair(*dist_list)
            out_unsort[(i,j)] = out_unsort[(j,i)] = dist
        for i in range(L+1):
            out_unsort[(i,i)] = Distance.Pair(*(10*[0]))
        pairs = OrderedDict({key: out_unsort[key] for key in sorted(out_unsort)})
        return cls(pairs)
