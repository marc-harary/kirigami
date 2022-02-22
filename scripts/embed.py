import torch
from tqdm import tqdm
from glob import glob
import os
from torch.utils.data import TensorDataset, IterableDataset, DataLoader
import numpy as np

BASES = {"A": 0, "C": 1, "G": 2, "U": 3}

def embed_bpseq(path, full_len=512):
    with open(path, "r") as f:
        lines = f.read().splitlines()
    seq = ""
    second = {}
    line_idx = 0
    con_idxs = [] 
    for line in lines:
        words = line.split()
        base = words[1] 
        if not base.isalpha():
            break
        seq += base
        ii, jj = int(words[0])-1, int(words[-1])-1
        if jj > -1:
            con_idxs.append((ii, jj))
    fasta = torch.zeros(4, len(seq), dtype=torch.uint8)
    idxs = [BASES[char] for char in seq]
    fasta[idxs, list(range(len(seq)))] = 1
    con = torch.zeros(len(seq), len(seq), dtype=torch.uint8)
    if len(con_idxs) > 0:
        con_idxs_ = torch.tensor(con_idxs)# + offset
        con[con_idxs_[:,0], con_idxs_[:,1]] = 1
        con[con_idxs_[:,1], con_idxs_[:,0]] = 1
    return con.to_sparse(), fasta.to_sparse(), seq


def embed_thermo(path, seq):
    out = torch.zeros(len(seq), len(seq))
    with open(path, "r") as f:
        txt = f.read()
        if txt == "":
            return out
        lines = txt.splitlines()
    idxs = []
    dists = []
    for line in lines:
        words = line.split()
        idxs.append(list(map(int, words[:2])))
        dists.append(float(words[-1]))
    idxs_t = torch.tensor(idxs, dtype=int)
    dists_t = torch.tensor(dists)
    idxs_t -= 1
    out[idxs_t[:,0], idxs_t[:,1]] = dists_t
    out[idxs_t[:,1], idxs_t[:,0]] = dists_t
    return out.to_sparse()



def embed_dist(file,
               size: int,
               K: int = 1.,
               eps: float = 1e-8,
               tau: float = 2,
               ceiling: float = 50.):
    with open(file, "r") as f:
        lines = f.read().splitlines()
    out = torch.zeros(10, size, size)
    for line in lines[1:]:
        words = line.split()
        ii, jj = int(words[2])-1, int(words[5])-1
        dists_ = list(map(float, words[6:]))
        dists = torch.tensor(dists_) 
        out[:,ii,jj] = out[:,jj,ii] = dists
    out = out.clip(out.min(), ceiling)
    out /= ceiling
    # out = out ** tau
    # out = K / (out + eps)
    return out#.to_sparse()


# def embed_dist(file,
#                size: int,
#                val_max: float = 30):
#     with open(file, "r") as f:
#         lines = f.read().splitlines()
#     out = torch.zeros(10, size, size)
#     for line in lines[1:]:
#         words = line.split()
#         ii, jj = int(words[2])-1, int(words[5])-1
#         dists_ = list(map(float, words[6:]))
#         dists = torch.tensor(dists_) 
#         out[:,ii,jj] = out[:,jj,ii] = dists
#     out = out.clip(0, val_max)
#     out /= val_max
#     # out = A / (out + eps)
#     return out#.to_sparse()
        
        
def main():
    bpseq_dir = "/gpfs/ysm/project/pyle/mah258/spot/pdb/TR1-bpseq"
    ct_dir = "/gpfs/ysm/project/pyle/mah258/spot/pdb/TR1-ct"
    dist_dir = "/home/mah258/project/spot/pdb/TR1-dists"
    bpseqs = sorted(glob(os.path.join(bpseq_dir, "*")))
    cts = sorted(glob(os.path.join(ct_dir, "*")))
    dists = sorted(glob(os.path.join(dist_dir, "*")))

    data = []
    files_zip = list(zip(bpseqs, cts, dists))
    for bpseq, ct, dist in tqdm(files_zip):
        con, seq, seq_str = embed_bpseq(bpseq)
        thermo = embed_thermo(ct, seq_str)
        dists = embed_dist(dist, len(seq_str))
        data.append((seq, thermo, con, dists))

    torch.save(data, "TR1_norm_clip50.pt")

    bpseq_dir = "/gpfs/ysm/project/pyle/mah258/spot/pdb/VL1-bpseq"
    ct_dir = "/gpfs/ysm/project/pyle/mah258/spot/pdb/VL1-ct"
    dist_dir = "/home/mah258/project/spot/pdb/VL1-dists"
    bpseqs = sorted(glob(os.path.join(bpseq_dir, "*")))
    cts = sorted(glob(os.path.join(ct_dir, "*")))
    dists = sorted(glob(os.path.join(dist_dir, "*")))

    data = []
    files_zip = list(zip(bpseqs, cts, dists))
    for bpseq, ct, dist in tqdm(files_zip):
        con, seq, seq_str = embed_bpseq(bpseq)
        thermo = embed_thermo(ct, seq_str)
        dists = embed_dist(dist, len(seq_str))
        data.append((seq, thermo, con, dists))

    torch.save(data, "VL1_norm_clip_50.pt")
    

if __name__ == "__main__":
    main()
