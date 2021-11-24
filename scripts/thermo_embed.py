import torch
from tqdm import tqdm
from glob import glob
import os
from torch.utils.data import TensorDataset

# BASES = {0: "A", 1: "U", 2: "G", 3: "C"}
BASES = {"A": 0, "U": 1, "G": 2, "C": 3}

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
        con_idxs.append((ii, jj))
        base += words[1]
        if jj != -1:
            second[ii] = jj 
        line_idx += 1

    offset = (full_len - len(seq)) // 2

    fasta = torch.zeros(4, full_len, dtype=torch.uint8)
    idxs = [BASES[char] for char in seq]
    fasta[idxs, list(range(offset, offset+len(seq)))] = 1

    con = torch.zeros(512, 512, dtype=torch.uint8)
    con_idxs_ = torch.tensor(con_idxs) + offset
    con[con_idxs_[:,0], con_idxs_[:,1]] = 1
    con[con_idxs_[:,1], con_idxs_[:,0]] = 1

    return con.to_sparse(), fasta.to_sparse(), offset


def embed_thermo(path, offset, full_len=512):
    with open(path, "r") as f:
        lines = f.read().splitlines()

    idxs = []
    dists = []
    for line in lines:
        words = line.split()
        idxs.append(list(map(int, words[:2])))
        dists.append(float(words[-1]))
    
    idxs_t = torch.tensor(idxs, dtype=int)
    dists_t = torch.tensor(dists)
   
    idxs_t += offset - 1
    out = torch.zeros(full_len, full_len)
    out[idxs_t[:,0], idxs_t[:,1]] = dists_t
    out[idxs_t[:,1], idxs_t[:,0]] = dists_t

    return out.to_sparse()
 
        

def main():
    bpseq_dir = "/home/mah258/project/spot/bpRNA/TR0-bpseq-cleaned"
    ct_dir = "/home/mah258/project/spot/bpRNA/TR0-cleaned-cts"
    
    bpseqs = glob(os.path.join(bpseq_dir, "*"))
    bpseqs.sort()

    cts = glob(os.path.join(ct_dir, "*"))
    cts.sort()

    cons_ = []
    seqs_ = []
    thermos_ = []
    files_zip = list(zip(bpseqs, cts))
    for bpseq, ct in tqdm(files_zip):
        con, seq, offset = embed_bpseq(bpseq)
        thermo = embed_thermo(ct, offset)
        seqs_.append(seq)
        thermos_.append(thermo)
        cons_.append(con)
    seqs = torch.stack(seqs_)
    thermos = torch.stack(thermos_)
    cons = torch.stack(cons_)
    dset = TensorDataset(seqs, thermos, cons)

    torch.save(dset, "TR0-thermo.pt")
    

if __name__ == "__main__":
    main()
