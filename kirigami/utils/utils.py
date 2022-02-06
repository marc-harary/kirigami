from typing import *
import torch


__all__ = ["concat", "collate_fn", "get_scores", "fasta2str", "prd2set", "grd2set", "get_dists"]
PAIRS = {"AU", "UA", "CG", "GC", "GU", "UG"}



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
    rho = cov / (prd.std()*grd.std())
    return rho


def collate_fn(batch_list: Sequence[Tuple],
               device: torch.device,
               use_thermo: bool,
               n_dists: Optional[int] = 10,
               use_dist: bool = False): 
    """Collates list of tensors in batch"""

    lengths = torch.tensor([tup[0].shape[1] for tup in batch_list])
    pad = max(lengths)
    seqs_, thermos_, cons_, dists_ = [], [], [], []

    for i, tup in enumerate(batch_list):
        seq_ = tup[0]
        offset = (pad.item() - seq_.shape[1]) // 2
        seq_idxs = seq_.coalesce().indices()
        seq_idxs[1,:] += offset
        seq = torch.zeros(4, pad, device=device)
        seq[seq_idxs[0,:], seq_idxs[1,:]] = 1
        seq = concat(seq)
        seqs_.append(seq)

        if use_thermo:
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

        if use_dist:
            dist_ = tup[3]
            # dist = dist_.to_dense().to(device)
            dist = dist_.to(device)
            dist = dist[:n_dists,:,:]
            dists_.append(dist)
            
    seqs = torch.stack(seqs_)
    cons = torch.stack(cons_)
    if use_thermo:
        thermos = torch.stack(thermos_)
        ipts = torch.cat((seqs, thermos), 1)
    else:
        ipts = seqs
    if use_dist:
        dists = torch.stack(dists_)
        opts = (cons, dists)
    else:
        opts = cons

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


def get_dists(prd: torch.Tensor,
              grd: torch.Tensor,
              inv: bool = False,
              eps: float = 1e-8,
              tau: float = 1,
              K: float = 100.,
              ceiling: float = None):
    prd_ = prd[1]
    grd_ = grd[1]
    L = prd_.shape[-1]

    # prd_dists = max_value * prd_
    # grd_dists = max_value * grd_

    prd_vec_ = []
    grd_vec_ = []
    for i in range(L):
        for j in range(i+4, L):
            idxs = grd_[:,:,i,j] > 0
            prd_vec_.append(prd_[:,:,i,j][idxs])
            grd_vec_.append(grd_[:,:,i,j][idxs])

    grd_vec = torch.hstack(grd_vec_).squeeze()
    prd_vec = torch.hstack(prd_vec_).squeeze()

    # if inv:
    #     grd_vec = K/grd_vec - eps
    #     prd_vec = K/prd_vec - eps
    #     # prd_vec = prd_vec ** 1/tau
    #     # grd_vec = grd_vec ** 1/tau

    # grd_vec *= 20
    # prd_vec *= 20
    prd_vec *= ceiling
    grd_vec *= ceiling

    grd_vec_sort = grd_vec.sort().values
    prd_vec_sort = prd_vec[grd_vec.sort().indices]
    error = torch.abs(prd_vec_sort - grd_vec_sort)
    
    pccs = []
    errors = []
    for num in [L, 2*L, 5*L, 10*L]:
        grd_vec_trunc = grd_vec_sort[:num]
        prd_vec_trunc = prd_vec_sort[:num]
        pccs.append(get_pcc(grd_vec_trunc, prd_vec_trunc))
        errors.append(error[:num].mean())
    pccs.append(get_pcc(grd_vec, prd_vec))
    errors.append(error.mean())

    return torch.Tensor(errors + pccs)
