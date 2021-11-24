from typing import *
from dataclasses import dataclass
import torch


class Contact(torch.Tensor):

    @dataclass
    class Scores:
        tp: float
        tn: float
        fp: float
        fn: float
        f1: float
        mcc: float
        ground_pairs: int
        pred_pairs: int

    @classmethod
    def from_dict(cls,
                  pairs: Dict[int,int],
                  length: Optional[int] = None,
                  dtype: torch.dtype = torch.uint8,
                  device: torch.device = torch.device("cpu")) -> "Contact":
        # assert dim >= 2
        native_length = max(max(pairs.keys()), max(pairs.values())) + 1
        length = max(length, native_length) if length else native_length
        size = (length, length)
        # size = (dim-len(size))*(1,) + size
        offset = (length - native_length) // 2
        out = torch.zeros(size, dtype=dtype, device=device)
        items = list(pairs.items())
        items_flipped = list(map(lambda tup: tup[::-1], items))
        idxs = offset + torch.tensor(items + items_flipped)
        out[...,idxs[:,0],idxs[:,1]] = 1.
        return out.as_subclass(cls)

    def to_dict(self, length: Optional[int] = None) -> Dict[int,int]:
        mat_ = self.squeeze()
        length = length or mat_.shape[0]
        values, js = torch.max(mat_, 0)
        items_tensor = torch.vstack((torch.arange(len(js)), js)).T.int()
        items_tensor = items_tensor[values == 1]
        items_list = items_tensor.tolist()
        items_list.sort()
        contact_map = OrderedDict(items_list)
        return contact_map

    def get_scores(self, ground: "Contact") -> Scores: 
        """returns various evaluative scores of predicted secondary structure"""
        assert (length := len(self)) == len(ground_map)
        total = length * (length-1) / 2
        pred_set = set(self._pairs.items())
        ground_set = set(ground._pairs.items())
        pred_pairs, ground_pairs = len(pred_set), len(ground_set)
        tp = float(len(pred_set.intersection(ground_set)))
        fp = len(pred_set) - tp
        fn = len(ground_set) - tp
        tn = total - tp - fp - fn
        mcc = f1 = 0. 
        if len(pred_set) != 0 and len(ground_set) != 0:
            sn = tp / (tp+fn)
            pr = tp / (tp+fp)
            if tp > 0:
                f1 = 2*sn*pr / (pr+sn)
            if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
                mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
        return Scores(tp, tn, fp, fn, f1, mcc, ground_pairs, pred_pairs)
