import torch
import torch.nn as nn
from collections import namedtuple


class PostProcess(nn.Module):
    def __init__(self):
        super().__init__()
        self._bases = torch.Tensor([2, 3, 5, 7])
        self._pairs = {14, 15, 35}
        self._min_dist = 4
        

    def forward(self, con, feat, ground_truth=torch.inf):
        # remove sharp angles
        con_ = con.squeeze()
        # con_ = con_.triu(self._min_dist+1) + con_.tril(-(self._min_dist+1))
        con_ = con_.triu(self._min_dist + 1)
        L = len(con_)
        
        # canonicalize
        seq = feat.squeeze()[:len(self._bases),:,0]
        pairs = self._bases.to(seq.device)[seq.argmax(0)]
        pair_mat = pairs.outer(pairs)
        pair_mask = torch.zeros(con_.shape, dtype=bool, device=con_.device)
        for pair in self._pairs:
            pair_mask = torch.logical_or(pair_mask, pair_mat == pair)
        con_[~pair_mask] = 0.

        if not self.training:
            # filter for maximum one pair per base
            con_flat = con_.flatten()
            idxs = con_flat.argsort(descending=True)
            ii = idxs % L
            jj = torch.div(idxs, L, rounding_mode="floor")
            ground_truth = min(ground_truth, L // 2)
            memo = torch.zeros(L, dtype=bool)
            one_mask = torch.zeros(L, L, dtype=bool)
            num_pairs = 0
            for i, j in zip(ii, jj):
                if num_pairs > ground_truth:
                    break
                if memo[i] or memo[j]:
                    continue
                one_mask[i, j] = one_mask[j, i] = True
                memo[i] = memo[j] = True
                num_pairs += 1
            con_[~one_mask] = 0.

        con_ = con_ + con_.transpose(-2, -1)
        con_ = con_.reshape_as(con)

        return con_


class ForkHead(nn.Module):
    def __init__(self, n_channels, n_bins, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.bin_conv = nn.Conv2d(in_channels=n_channels,
                                  out_channels=n_bins,
                                  kernel_size=kernel_size,
                                  padding=padding)
        self.inv_conv = nn.Conv2d(in_channels=n_channels,
                                  out_channels=1,
                                  kernel_size=kernel_size,
                                  padding=padding)

    def forward(self, ipt):
        opt = {}
        opt["bin"] = self.bin_conv(ipt)
        opt["bin"] = (opt["bin"] + opt["bin"].transpose(-1, -2)) / 2
        opt["inv"] = self.inv_conv(ipt).sigmoid()
        opt["inv"] = (opt["inv"] + opt["inv"].transpose(-1, -2)) / 2
        return opt
        

class Fork(nn.Module):

    dist_types = ["PP", "O5O5", "C5C5", "C4C4", "C3C3", "C2C2", "C1C1", "O4O4", "O3O3", "NN"]
    Opt = namedtuple("Opt", ["con"] + dist_types)

    def __init__(self, n_channels, n_bins, kernel_size):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.con_conv = nn.Conv2d(in_channels=n_channels,
                                  out_channels=1,
                                  kernel_size=kernel_size,
                                  padding=padding)
        for dist_type in self.dist_types:
            head = ForkHead(n_channels, n_bins, kernel_size)
            setattr(self, dist_type, head)

    def forward(self, ipt):
        opt = {}
        opt["dists"] = {}
        opt["con"] = self.con_conv(ipt).sigmoid()
        opt["con"] = (opt["con"] + opt["con"].transpose(-1, -2)) / 2
        for dist_type in self.dist_types:
            head = getattr(self, dist_type)
            opt["dists"][dist_type] = head(ipt)
        return opt
