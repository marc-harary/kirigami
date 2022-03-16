from typing import *
import torch


__all__ = ["concat", "collate_fn", "get_scores", "fasta2str", "prd2set", "grd2set", "get_dists", "unembed"]
PAIRS = {"AU", "UA", "CG", "GC", "GU", "UG"}
NUM_DISTS = 10



def concat(fasta):
    """Outer concatenation on sequence tensor""" 
    out = fasta.unsqueeze(-1)
    out = torch.cat(out.shape[-2] * [out], dim=-1)
    out_t = out.transpose(-1, -2)
    out = torch.cat([out, out_t], dim=-3)
    return out


def get_pcc(prd, grd):
    prd_demean = prd - prd.mean()
    grd_demean = grd - grd.mean()
    cov = torch.mean(prd_demean * grd_demean)
    if prd.std() == 0 or grd.std() == 0:
        return 0
    return cov / (prd.std()*grd.std())


def one_hot_bin(ipt, bins):
    n_bins = bins.numel()
    n_data = ipt.numel()
    # expand both tensors to shape (ipt.size(), bins.size())
    ipt_flat = ipt.flatten()
    ipt_exp = ipt_flat.expand(n_bins, -1)
    bins_exp = bins.expand(n_data, -1).T
    # find which bins ipt fits in
    # TODO: do less or equal to 
    bin_bools = (ipt_exp <= bins_exp).int()
    # get index of largest bin ipt fits in
    vals, idxs = torch.max(bin_bools, 0)
    # if max is 0, then val is greater than every bin
    idxs[vals == 0] = n_bins-1
    # construct one-hot
    one_hot = torch.zeros(n_bins, n_data, device=ipt.device)
    one_hot[idxs, torch.arange(n_data)] = 1
    # reshape back into ipt's shape
    one_hot = one_hot.reshape(n_bins, *ipt.shape)
    return one_hot



def collate_fn(batch_list: Sequence[Tuple],
               device: torch.device,
               use_thermo: bool = True,
               use_dist: bool = False,
               multiclass: bool = False,
               bins: Optional[torch.Tensor] = None,
               dist_idxs: Optional[List[int]] = None,
               ceiling: float = torch.inf,
               inv: bool = False,
               inv_eps: float = 1e-8):
    """Collates list of tensors in batch"""
    assert not inv or not multiclass

    lengths = torch.tensor([tup[0].shape[1] for tup in batch_list])
    pad = max(lengths)
    seqs_, thermos_, cons_, dists_ = [], [], [], []
    # dists_ = [[] for dist_idx in dist_idxs]
    # TODO: clean up this line
    dists_ = []

    dist_idxs = dist_idxs if dist_idxs else torch.arange(10)

    for i, tup in enumerate(batch_list):
        seq_ = tup[0]
        offset = (pad.item() - seq_.shape[1]) // 2
        seq_idxs = seq_.coalesce().indices()
        seq_idxs[1,:] += offset
        seq = torch.zeros(4, pad, device=device)
        seq[seq_idxs[0,:], seq_idxs[1,:]] = 1
        seq = concat(seq)
        seqs_.append(seq)

        thermo_ = tup[1]
        # error handling accounts for what's probably a PyTorch bug
        try:
            thermo_idxs = thermo_.coalesce().indices()
        except NotImplementedError:
            thermo_idxs = torch.tensor([[], []], dtype=torch.long)
        thermo = torch.zeros(1, pad, pad, device=device)
        thermo[0, thermo_idxs[0,:], thermo_idxs[1,:]] = 1
        thermo_idxs += offset
        thermos_.append(thermo)

        con_ = tup[2]
        con_idxs = con_.coalesce().indices()
        con_idxs += offset
        con = torch.zeros(1, pad, pad, device=device)
        con[0, con_idxs[0,:], con_idxs[1,:]] = 1
        cons_.append(con)

        dist_ = tup[3]
        dist = dist_.to(device)
        dist = dist.clip(-torch.inf, ceiling)
        if inv:
            dist_out = 1 / (dist + inv_eps)
            dist_out[dist <= 0] = torch.nan
        elif multiclass:
            dist_out = one_hot_bin(dist, bins)
            dist_out[:, dist <= 0] = torch.nan
        else:
            dist_out = dist
            dist_out[dist <= 0] = torch.nan
        dists_.append(dist_out)
            
    seqs = torch.stack(seqs_)
    thermos = torch.stack(thermos_)
    cons = torch.stack(cons_)
    dists_stack = torch.stack(dists_)
    dists_stack = dists_stack[:,:,dist_idxs,:,:]
    dists = torch.tensor_split(dists_stack, len(dist_idxs), dim=-3)
    dists = [dist.squeeze(-3) for dist in dists]

    ipts = seqs
    if use_thermo:
        ipts = torch.cat((ipts, thermos), 1)
    opts = cons
    if use_dist:
        opts = (opts, *dists)

    return ipts, opts



def fasta2str(ipt: torch.Tensor) -> str:
    """Converts `Tensor` to FASTA string"""
    if isinstance(ipt, Tuple):
        ipt_ = ipt[0]
    else:
        ipt_ = ipt 
    ipt_ = ipt.squeeze()[:4,:,0].char()
    total_length = ipt_.shape[1]
    fasta_length = int(ipt_.sum().item())
    _, js = torch.max(ipt_, 0)
    return "".join("ACGU"[j] for j in js)



def get_scores(prd_pairs: set, grd_pairs: set, seq_len: int) -> dict:
    """Calculates scores on pairs"""
    total = seq_len * (seq_len-1) / 2
    n_prd, n_grd = len(prd_pairs), len(grd_pairs)
    tp = float(len(prd_pairs.intersection(grd_pairs)))
    fp = len(prd_pairs) - tp
    fn = len(grd_pairs) - tp
    tn = total - tp - fp - fn
    mcc = f1 = 0. 
    if n_prd > 0 and n_grd > 0:
        sn = tp / (tp+fn)
        pr = tp / (tp+fp)
        if tp > 0:
            f1 = 2*sn*pr / (pr+sn)
        if (tp+fp) * (tp+fn) * (tn+fp) * (tn+fn) > 0:
            mcc = (tp*tn-fp*fn) / ((tp+fp)*(tp+fn)*(tn+fp)*(tn+fn))**.5
    return {"tp": tp,
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "f1": f1,
            "mcc": mcc,
            "n_grd": n_grd,
            "n_prd": n_prd}


def grd2set(ipt: torch.Tensor):
    """Converts ground truth contact `Tensor` to set of ordered pairs"""
    if isinstance(ipt, Tuple):
        ipt_ = ipt[0]
    else:
        ipt_ = ipt 
    ipt_ = ipt_.squeeze()
    idxs = torch.where(ipt_) 
    idxs = torch.vstack(idxs).T
    out = set()
    for i, j in idxs:
        if i < j:
            out.add((i.item(), j.item()))
    return out


def prd2set(ipt: torch.Tensor,
            thres_pairs: int,
            seq: str,
            min_dist: int = 4,
            min_prob: float = .5, # TODO implement
            canonical: bool = True): # TODO implement
    """Converts predicted contact `Tensor` to set of ordered pairs"""
    if isinstance(ipt, Tuple):
        ipt_ = ipt[0]
    else:
        ipt_ = ipt 
    ipt_ = ipt_.squeeze() 
    ipt_ = (ipt_ + ipt_.T) / 2 # symmetrize matrix
    # get indices sorted in descending order by probability
    ipt_ = ipt_.flatten() # PyTorch makes sorting 2D tensors impractical
    idxs = ipt_.argsort(descending=True)
    # convert 1D index to 2D index using pointer arithmetic
    ii = idxs % len(seq)
    jj = torch.div(idxs, len(seq), rounding_mode="floor")
    kept = torch.zeros(len(seq), dtype=bool) # records indices already used
    out_set = set()
    num_pairs = 0
    for (i, j) in zip(ii, jj):
        if num_pairs == thres_pairs:
            break
        if (seq[i]+seq[j] in PAIRS and # canonical base pairs
            not kept[i] and not kept[j] and # not already
            i < j and # get upper triangular matrix
            j - i >= min_dist): # ensure i and j are at least min_dist apart
                out_set.add((i.item(), j.item()))
                kept[i] = kept[j] = True
                num_pairs += 1
    return out_set


# def get_dists(prd: torch.Tensor,
#               grd: torch.Tensor,
#               A: float = 1.,
#               eps: float = 1e-8):
#     prd_ = prd[1]
#     grd_ = grd[1]
#     L = prd_.shape[-1]
# 
#     prd_dists = A / prd_ - eps
#     grd_dists = A / grd_ - eps
# 
#     prd_vec_ = []
#     grd_vec_ = []
#     for i in range(L):
#         for j in range(i+4, L):
#             prd_vec_.append(prd_dists[:,:,i,j])
#             grd_vec_.append(grd_dists[:,:,i,j])
# 
#     prd_vec = torch.hstack(prd_vec_)
#     grd_vec = torch.hstack(grd_vec_)
#     prd_vec = prd_vec.sort().values
#     grd_vec = grd_vec.sort().values
# 
#     error = torch.abs(prd_vec - grd_vec)
#         
#     return torch.Tensor([error[:,:L].mean(),
#                          error[:,:2*L].mean(),
#                          error[:,:5*L].mean(),
#                          error[:,:10*L].mean()])


def unembed(ipt: torch.Tensor,
             bins: torch.Tensor,
             ceiling: float = None,
             inv: bool = False,
             inv_eps: float = 1e-8,
             multiclass: bool = False):
    assert not multiclass or not inv
    if isinstance(ipt, tuple):
        return tuple([unembed(tens, bins, ceiling, inv, inv_eps, multiclass) for tens in ipt])
    elif inv:
        return 1/ipt - inv_eps
    elif multiclass:
        out = torch.zeros(ipt.shape[-1], ipt.shape[-1])
        # bin_list = [0] + bins.tolist()
        bin_centers = torch.zeros(len(bins)+1)
        for i in range(len(bins)-1):
            bin_centers[i] = 0.5 * (bins[i] + bins[i+1])
        bin_centers[-1] = bins[-1]
        ipt_idxs = ipt.argmax(1)
        out = bin_centers[ipt_idxs]
        return out


def get_dists(prd: torch.Tensor,
              grd: torch.Tensor):
              # bins: torch.Tensor, 
              # ceiling: float = None,
              # inv: bool = False,
              # eps: float = 1e-8):
              # tau: float = 1,
              # K: float = 100.,
              # ceiling: float = None):
    prd_ = prd.squeeze()
    grd_ = grd.squeeze()
    L = prd_.shape[-1]

    prd_vec_ = []
    grd_vec_ = []
    for i in range(L):
        for j in range(i+4, L):
            idxs = grd_[i,j] > 0
            prd_vec_.append(prd_[i,j][idxs])
            grd_vec_.append(grd_[i,j][idxs])

    grd_vec = torch.hstack(grd_vec_).squeeze()
    prd_vec = torch.hstack(prd_vec_).squeeze()

    # prd_vec *= ceiling
    # grd_vec *= ceiling

    grd_vec_sort = grd_vec.sort().values
    prd_vec_sort = prd_vec[grd_vec.sort().indices]
    error = torch.abs(prd_vec_sort - grd_vec_sort)
    
    l_pccs = []
    l_errors = []
    for num in [L, 2*L, 5*L, 10*L]:
        grd_vec_trunc = grd_vec_sort[:num]
        prd_vec_trunc = prd_vec_sort[:num]
        l_pccs.append(get_pcc(grd_vec_trunc, prd_vec_trunc))
        l_errors.append(error[:num].mean())
    l_pccs.append(get_pcc(grd_vec, prd_vec))
    l_errors.append(error.mean())

    d_errors = [] 
    # d_errors = [] 
    # for i in range(len(bins) - 1):
    #     lower = bins[i]
    #     upper = bins[i+1]
    #     mask = torch.logical_and(grd_vec_sort > lower,
    #                              grd_vec_sort < upper)
    #     d_error = 0
    #     if any(mask):
    #         grd_vec_trunc = grd_vec_sort[mask]
    #         error_trunc = error[mask]
    #         d_error = error_trunc.mean()
    #     d_errors.append(d_error)
        
    return torch.Tensor(l_pccs), torch.Tensor(l_errors), torch.Tensor(d_errors)
