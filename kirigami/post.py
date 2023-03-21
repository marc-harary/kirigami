import numpy as np
import torch
from torch import nn
import networkx as nx

import pyximport
pyximport.install(setup_args=dict(include_dirs=np.get_include()))
import kirigami.nussinov


class Symmetrize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ipt):
        return (ipt + ipt.transpose(-2, -1)) / 2


class RemoveSharp(nn.Module):
    def __init__(self):
        super().__init__()
        self._min_dist = 4

    def forward(self, ipt):
        return ipt.tril(-self._min_dist) + ipt.triu(self._min_dist)


class Canonicalize(nn.Module):
    def __init__(self):
        super().__init__()
        self._bases = torch.Tensor([2, 3, 5, 7])
        self._pairs = {14, 15, 35}

    def forward(self, con, feat):
        con_ = con.squeeze()
        seq = feat.squeeze()[:len(self._bases),:,0]
        pairs = self._bases.to(seq.device)[seq.argmax(0)]
        pair_mat = pairs.outer(pairs)
        pair_mask = torch.zeros(con_.shape, dtype=bool, device=con_.device)
        for pair in self._pairs:
            pair_mask = torch.logical_or(pair_mask, pair_mat == pair)
        con_[~pair_mask] = 0.
        return con_.reshape_as(con)
        


class Greedy(nn.Module):
    def __init__(self):
        super().__init__()
        self._bases = torch.Tensor([2, 3, 5, 7])
        self._pairs = {14, 15, 35}
        self._min_dist = 4
        self.symmetrize = Symmetrize()
        self.remove_sharp = RemoveSharp()
        self.canonicalize = Canonicalize()
        
    def forward(self, con, feat, ground_truth=torch.inf, sym_only=False):

        con = self.symmetrize(con)

        if self.training or sym_only:
            return con
        
        con = self.remove_sharp(con)
        con = self.canonicalize(con, feat)
        con_ = con.squeeze()

        # filter for maximum one pair per base
        L = len(con_)
        con_flat = con_.flatten()
        idxs = con_flat.argsort(descending=True)
        ii = idxs % L
        jj = torch.div(idxs, L, rounding_mode="floor")
        ground_truth = min(ground_truth, L // 2)
        memo = torch.zeros(L, dtype=bool)
        one_mask = torch.zeros(L, L, dtype=bool)
        num_pairs = 0
        for i, j in zip(ii, jj):
            if num_pairs == ground_truth:
                break
            if memo[i] or memo[j]:
                continue
            one_mask[i, j] = one_mask[j, i] = True
            memo[i] = memo[j] = True
            num_pairs += 1
        con_[~one_mask] = 0.

        return con_.reshape_as(con)


class Dynamic(nn.Module):
    def __init__(self):
        super().__init__()
        self.symmetrize = Symmetrize()
        self.remove_sharp = RemoveSharp()
        self.canonicalize = Canonicalize()

    def forward(self, con, feat, *args, **kwargs):
        if self.training:
            return con

        con = self.symmetrize(con)
        con = self.remove_sharp(con)
        con = self.canonicalize(con, feat)
        con_ = con.squeeze()

        con_np = con_.cpu().numpy()
        pair_mask_np = kirigami.nussinov.nussinov(con_np.astype(np.float64))
        pair_mask = torch.from_numpy(pair_mask_np).to(con.device)
        con_ *= pair_mask

        return con_.reshape_as(con)


class Blossom(nn.Module):
    def __init__(self):
        super().__init__()
        self._bases = torch.Tensor([2, 3, 5, 7])
        self._pairs = {14, 15, 35}
        self._min_dist = 4
        self.symmetrize = Symmetrize()
        self.remove_sharp = RemoveSharp()
        self.canonicalize = Canonicalize()
        
    def forward(self, con, feat, ground_truth=torch.inf, sym_only=False):
        con = self.symmetrize(con)

        if self.training or sym_only:
            return con
        con = self.remove_sharp(con)
        con = self.canonicalize(con, feat)

        if torch.sum(con > 0) == 0:
            return con

        con_cpu = con.clone().cpu()
        con_cpu = con_cpu.squeeze()
        L = con_cpu.shape[-1]

        G = nx.Graph()
        edges = []
        for i in range(L):
            for j in range(L):
                if con_cpu[i, j] > 0:
                    edges.append((i, j, con_cpu[i, j]))
        G.add_weighted_edges_from(edges)
        edges_match = torch.tensor(list(nx.max_weight_matching(G)))

        mask = torch.zeros_like(con_cpu)
        mask[edges_match[:, 0], edges_match[:, 1]] = 1
        mask[edges_match[:, 1], edges_match[:, 0]] = 1
        mask = mask.to(con.device)
        mask = mask.reshape_as(con)
        con = con * mask

        return con

