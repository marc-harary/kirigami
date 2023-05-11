from typing import *
import sys
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from math import log, ceil
from kirigami.constants import *


class Symmetrize(nn.Module):
    """
    Symmetrizes tensor along last dimension.
    """

    def __init__(self):
        super().__init__()

    def forward(self, ipt):
        return (ipt + ipt.transpose(-1, -2)) / 2


class RemoveSharp(nn.Module):
    """
    Removes sharp angles from base-pairing probabilities.
    """

    def __init__(self):
        super().__init__()

    def forward(self, ipt):
        return ipt.tril(-MIN_DIST) + ipt.triu(MIN_DIST)


class Canonicalize(nn.Module):
    """
    Excludes non-canonical base pairs.
    """

    def __init__(self):
        super().__init__()

    def forward(self, con, feat):
        con_ = con.squeeze()
        seq = feat.squeeze()[: len(BASE_PRIMES), :, 0]
        pairs = BASE_PRIMES.to(seq.device)[seq.argmax(0)]
        pair_mat = pairs.outer(pairs)
        pair_mask = torch.zeros(con_.shape, dtype=bool, device=con_.device)
        for pair in PAIRS:
            pair_mask = torch.logical_or(pair_mask, pair_mat == pair)
        con_[~pair_mask] = 0.0
        return con_.reshape_as(con)


class Greedy(nn.Module):
    """
    Performs greedy post-processing.
    """

    def __init__(self):
        super().__init__()
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
        idxs = con_flat.cpu().argsort(descending=True)
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
        con_[~one_mask] = 0.0

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


class ResNetBlock(nn.Module):
    """
    Implements main residual neural network (RNN) block.

    Attributes
    ----------
    p : float
        Dropout probability.
    dilations : Tuple[int, int]
        Kernel dilations for each block.
    kernel_sizes : Tuple[int, int]
        Sizes of kernels in each block.
    n_channels : int
        Number of hidden channels per block.
    act: str
        Class name of non-linearity in each block.
    """

    def __init__(
        self,
        p: float,
        dilations: Tuple[int, int],
        kernel_sizes: Tuple[int],
        n_channels: int,
        act: str = "GELU",
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        p : float
            Dropout probability.
        dilations : Tuple[int, int]
            Kernel dilations for each block.
        kernel_sizes : Tuple[int, int]
            Sizes of kernels in each block.
        n_channels : int
            Number of hidden channels per block.
        act: str
            Class name of non-linearity in each block.
        """
        super().__init__()
        self.resnet = True  # resnet
        self.conv1 = torch.nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_sizes[0],
            dilation=dilations[0],
            padding=self.get_padding(dilations[0], kernel_sizes[0]),
            bias=True,
        )
        self.norm1 = torch.nn.InstanceNorm2d(n_channels)
        self.act1 = getattr(nn, act)()
        self.drop1 = torch.nn.Dropout(p=p)
        self.conv2 = torch.nn.Conv2d(
            in_channels=n_channels,
            out_channels=n_channels,
            kernel_size=kernel_sizes[1],
            dilation=dilations[1],
            padding=self.get_padding(dilations[1], kernel_sizes[1]),
            bias=True,
        )
        self.norm2 = nn.InstanceNorm2d(n_channels)
        self.act2 = getattr(nn, act)()

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ipt : torch.Tensor
            Input tensor corresponding to activation of previous block.
        """
        out = ipt
        out = self.conv1(out)
        out = self.norm1(out)
        out = self.act1(out)
        if self.training:
            out = self.drop1(out)
        out = self.conv2(out)
        out = self.norm2(out)
        out += ipt
        out = self.act2(out)
        # out = 0.5 * (out + out.transpose(-1, -2))
        return out

    @staticmethod
    def get_padding(dilation: int, kernel_size: int) -> int:
        return round((dilation * (kernel_size - 1)) / 2)


class ResNet(nn.Module):
    """
    Implements main residual neural network (RNN) module.

    Attributes
    ----------
    n_blocks : int
        Number of residual neural network blocks.
    n_channels : int
        Number of hidden channels per block.
    kernel_sizes : Tuple[int, int]
        Sizes of kernels in each block.
    dilations : Tuple[int, int]
        Kernel dilations for each block.
    activation: str
        Class name of non-linearity in each block.
    dropout : float = dropout
        Dropout probability.
    trunk : torch.nn.Sequential
        Container for input, hidden, and output layers
    """

    def __init__(
        self,
        n_blocks: int,
        n_channels: int,
        kernel_sizes: List[int],
        dilations: List[int],
        activation: str,
        dropout: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        n_blocks : int
            Number of residual neural network blocks.
        n_channels : int
            Number of hidden channels per block.
        kernel_sizes : Tuple[int, int]
            Sizes of kernels in each block.
        dilations : Tuple[int, int]
            Kernel dilations for each block.
        activation: str
            Class name of non-linearity in each block.
        dropout : float = dropout
            Dropout probability.
        """
        super().__init__()
        if dilations is None:
            dilations = n_blocks * [1]
        else:
            num_cycles = ceil(2 * n_blocks / len(dilations))
            dilations = num_cycles * dilations
        self.n_blocks = n_blocks
        self.n_channels = n_channels
        self.kernel_sizes = kernel_sizes
        self.dilations = dilations
        self.activation = activation
        self.dropout = dropout
        trunk_list = [
            nn.Conv2d(8, n_channels, kernel_size=1, padding=0),
            getattr(nn, activation)(),
        ]
        for i in range(n_blocks):
            block = ResNetBlock(
                p=dropout,
                dilations=dilations[2 * i : 2 * (i + 1)],
                kernel_sizes=kernel_sizes,
                n_channels=n_channels,
                act=activation,
            )
            trunk_list.append(block)
        trunk_list.append(nn.Conv2d(n_channels, 1, kernel_size=1))
        trunk_list.append(nn.Sigmoid())
        trunk_list.append(Symmetrize())
        self.trunk = nn.Sequential(*trunk_list)

    def forward(self, ipt: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        ipt : torch.Tensor
            Input tensor corresponding to embedded sequence.
        """
        return self.trunk(ipt.float())