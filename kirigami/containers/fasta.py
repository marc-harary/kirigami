from typing import *
from copy import copy
import torch


__all__ = ["Fasta"]


class Fasta(torch.Tensor):

    _char_dict = OrderedDict({"A": 0, "U": 1, "C": 2, "G": 3})
    _idx_dict = OrderedDict({0: "A", 1: "U", 2: "G", 3: "C"})
    _is_concat: bool

    def concat(self) -> torch.Tensor:
        """Performs outer concatenation on batch of fasta tensors. Implemented as
           static method to be reused outside `Contact` object in training 
           script"""
        out = self.unsqueeze(-1)
        out = torch.cat(out.shape[-2] * [out], dim=-1)
        out_t = out.transpose(-1, -2)
        out = torch.cat([out, out_t], dim=-3)
        out = out.as_subclass(type(self))
        out._is_concat = True
        return out 

    def deconcat(self) -> torch.Tensor:
        assert self._is_concat
        return self[:4,:,0].as_subclass(type(self)) 

    @property
    def is_concat(self) -> bool:
        return self._is_concat

    def to_str(self) -> str:
        out = self.deconcat() if self._is_concat else self.squeeze()
        total_length = out.shape[1]
        fasta_length = int(out.sum().item())
        beg = (total_length - fasta_length) // 2
        end = beg + fasta_length
        _, js = torch.max(out[:,beg:end], 0)
        return "".join([self._idx_dict[j.item()] for j in js])

    @classmethod
    def from_str(cls,
                 fasta: str,
                 length: Optional[int] = None,
                 dtype: torch.dtype = torch.uint8,
                 device: torch.device = torch.device("cpu")) -> "Fasta":
        # assert dim >= 2
        length = max(length, len(fasta)) if length else len(fasta)
        size = (4, length)
        # size = (dim-len(size))*(1,) + size
        out = torch.zeros(size, device=device, dtype=dtype)
        offset = (length - len(fasta)) // 2
        idxs = [Fasta._char_dict[char] for char in fasta]
        out[..., idxs, list(range(offset,offset+length))] = 1.
        out = out.as_subclass(cls)
        out._is_concat = False
        return out
