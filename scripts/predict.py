import argparse
import torch
from torch import nn
import numpy as np


BASES = {"A": 0, "C": 1, "G": 2, "U": 3}


class Symmetrize(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, ipt):
        out = []
        if isinstance(ipt, Tuple):
            out = tuple((((tens + torch.transpose(tens, -1, -2)) / 2) for tens in ipt))
        else:
            out = (ipt + torch.transpose(ipt, -1, -2)) / 2
        return out


def concat(fasta):
    """Outer concatenation on sequence tensor""" 
    out = fasta.unsqueeze(-1)
    out = torch.cat(out.shape[-2] * [out], dim=-1)
    out_t = out.transpose(-1, -2)
    out = torch.cat([out, out_t], dim=-3)
    return out


def embed_fasta(path):
    with open(path, "r") as f:
        txt = f.read()
    lines = txt.splitlines()
    seq = lines[1]
    fasta = torch.zeros(4, len(seq), dtype=torch.uint8)
    idxs = [BASES[char] for char in seq]
    fasta[idxs, list(range(len(seq)))] = 1
    return fasta, len(seq)


def embed_pet(file, L: int):
    with open(file, "r") as f:
        lines = f.read().splitlines()
    out = torch.zeros(L, L, dtype=torch.uint8)
    for line in lines[1:]:
        words = line.split()
        nt1, nt2 = words[0], words[4]
        ii, jj = int(nt1)-1, int(nt2)-1
        if ii < jj:
            out[ii,jj] = out[jj,ii] = 1
    return out


def main():
    parser = argparse.ArgumentParser(description="Predict interatomic distance.")
    parser.add_argument("-m", "--model", help="Path to model.")
    parser.add_argument("--disable-cuda", action="store_true", help="Disable CUDA.")
    parser.add_argument("-f", "--fasta", help="Path to FASTA file.")
    parser.add_argument("-p", "--pet", help="Path to .ct file from PETFold.")
    parser.add_argument("-d", "--dists", help="Path to output distance Numpy file.")
    parser.add_argument("-c", "--contact", help="Path to output contact Numpy file.")
    args = parser.parse_args()

    model = torch.load(args.model, map_location=torch.device("cpu"))[0]
    if torch.cuda.is_available() and not args.disable_cuda:
        device = torch.device("cuda")
        model.to(device)
    else:
        device = torch.device("cpu")
        
    seq, L = embed_fasta(args.fasta)
    seq = concat(seq).to(device)
    pet = embed_pet(args.pet, L).to(device)
    pet = pet.unsqueeze(0)
    ipt = torch.cat((seq, pet), dim=0)
    ipt = ipt.unsqueeze(0)
    ipt = ipt.float()
    
    out = model(ipt)
    out = tuple([tens.detach().cpu().numpy().squeeze() for tens in out])
    dists = np.stack(out[1:])
    con = out[0]

    np.save(args.dists, dists)
    np.save(args.contact, con)
    

if __name__ == "__main__":
    main()
