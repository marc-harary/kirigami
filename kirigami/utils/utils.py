import torch

__all__ = ["concat", "collate_fn", "get_scores", "fasta2str", "prd2set", "grd2set"]
PAIRS = {"AU", "UA", "CG", "GC", "GU", "UG"}


def concat(fasta):
    """Outer concatenation on sequence tensor""" 
    out = fasta.unsqueeze(-1)
    out = torch.cat(out.shape[-2] * [out], dim=-1)
    out_t = out.transpose(-1, -2)
    out = torch.cat([out, out_t], dim=-3)
    return out


def collate_fn(batch_list, device: torch.device):
    """Collates list of tensors in batch"""
    lengths = torch.tensor([tup[0].shape[1] for tup in batch_list])
    pad = max(lengths)
    seqs_, thermos_, cons_ = [], [], []
    sums = []
    for i, tup in enumerate(batch_list):
        seq_, thermo_, con_ = tup
        offset = (pad.item() - seq_.shape[1]) // 2

        seq_idxs = seq_.coalesce().indices()
        thermo_idxs = thermo_.coalesce().indices()
        con_idxs = con_.coalesce().indices()

        seq_idxs[1,:] += offset
        thermo_idxs += offset
        con_idxs += offset

        # seq[seq_idxs.T.tolist()] = 1
        seq = torch.zeros(4, pad, dtype=torch.uint8)
        seq[seq_idxs[0,:], seq_idxs[1,:]] = 1
        sums.append(seq.sum())
        seq = concat(seq)
        thermo = torch.zeros(pad, pad, dtype=torch.uint8)
        thermo[thermo_idxs[0,:], thermo_idxs[1,:]] = 1
        con = torch.zeros(pad, pad, dtype=torch.uint8)
        con[con_idxs[0,:], con_idxs[1,:]] = 1

        seqs_.append(seq.float())
        thermos_.append(thermo.float())
        cons_.append(con.float())

        seqs = torch.stack(seqs_).to(device)
        seqs.requires_grad = True
        thermos = torch.stack(thermos_).to(device)
        thermos.requires_grad = True
        cons = torch.stack(cons_).to(device)
        # cons.requires_grad = True

    return seqs, thermos, cons


def fasta2str(ipt: torch.Tensor) -> str:
    """Converts `Tensor` to FASTA string"""
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
    ipt = ipt.squeeze()
    idxs = torch.where(ipt) 
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
    ipt = ipt.squeeze() 
    ipt = (ipt + ipt.T) / 2 # symmetrize matrix
    # get indices sorted in descending order by probability
    ipt_ = ipt.flatten() # PyTorch makes sorting 2D tensors impractical
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
