import string
import os
import subprocess
import re
from collections import deque
import numpy as np
import torch
# import RNA
import matplotlib.pyplot as plt
from matplotlib import patches

import pyximport
# pyximport.install(setup_args=dict(include_dirs=np.get_include()))
# import nussinov


ROOT = os.path.dirname(__file__)
EXE = os.path.join(ROOT, "PETfold")
os.environ["PETFOLDBIN"] = ROOT

SPOT_EXE = "/home/mah258/SPOT-RNA/SPOT-RNA.py"
PROB_PAIRS_EXE = "/home/mah258/CSSR/exe/ProbablePairRR"

PSEUDO_LEFT = "({[<" + string.ascii_uppercase
PSEUDO_RIGHT = ")}]>" + string.ascii_lowercase
COLORS = dict(A="red", C="green", G="blue", U="yellow")


def apply_cons(dbn, seq_str):
    CANON = {"AU", "UA", "CG", "GC", "UG", "GU"}
    idxs = parsedb(dbn, return_idxs=True)
    idxs_kept = []
    for pair in idxs:
        i, j = min(pair), max(pair)
        if seq_str[i] + seq_str[j] in CANON and j - i >= 4:
            idxs_kept.append((i, j))
    out_dbn = dict2db(idxs_kept, len(seq_str))
    return out_dbn, parsedb(out_dbn)


def build_table(pair_dict, L):
    memo = np.zeros((L, L))
    for k in range(1, L):
        for i in range(L - k):
            j = i + k
            unpairi = memo[i+1, j]
            unpairj = memo[i, j-1]
            pairij = memo[i+1, j-1] + int((i, j) in pair_dict or (j, i) in pair_dict)
            bifurc = 0
            for l in range(i, j):
                bifurc = max(bifurc, memo[i, l] + memo[l+1, j])
            memo[i, j] = max(unpairi, unpairj, pairij, bifurc)
    return memo


def trace_table(memo, pairs = None, i = None, j = None):
    if pairs is None:
        pairs = {}
        L = memo.shape[0]
        return trace_table(memo, pairs, 0, L-1)
    elif i >= j:
        return pairs
    elif memo[i, j] == memo[i+1, j]:
        return trace_table(memo, pairs, i+1, j)
    elif memo[i, j] == memo[i, j-1]:
        return trace_table(memo, pairs, i, j-1)
    elif memo[i, j] == memo[i+1, j-1] + 1: # canon(seq[i] + seq[j]):
        pairs[i] = j
        pairs[j] = i
        return trace_table(memo, pairs, i+1, j-1)
    else:
        for k in range(i+1, j-1):
            if memo[i, j] == memo[i, k] + memo[k+1, j]:
                trace_table(memo, pairs, i, k)
                trace_table(memo, pairs, k+1, j)
                break
    return pairs


def parsedb(dbn, return_idxs=False, split_pseudo=False):
    stacks = tuple((deque() for _ in PSEUDO_LEFT))
    nest_idxs = []
    pseudo_idxs = []
    for i, char in enumerate(dbn):
        if char in PSEUDO_LEFT:
            stacks[PSEUDO_LEFT.index(char)].append(i)
        elif char in PSEUDO_RIGHT:
            try:
                j = stacks[PSEUDO_RIGHT.index(char)].pop()
                if char == ")":
                    nest_idxs.append((i,j))
                    nest_idxs.append((j,i))
                else:
                    pseudo_idxs.append((i,j))
                    pseudo_idxs.append((j,i))
            except IndexError:
                continue
    if return_idxs:
        return nest_idxs, pseudo_idxs
    else:
        out = torch.zeros(len(dbn), len(dbn))
        for i, j in nest_idxs + pseudo_idxs:
            out[i, j] = 1
        return out.int()


def dict2db(idxs, L):
    pair_hierarch = []
    pairs = set(idxs)
    while pairs:
        memo = build_table(pairs, L)
        pairs_nest = trace_table(memo)
        pairs_nest = set(pairs_nest.items())
        pairs -= pairs_nest
        pair_hierarch.append(pairs_nest)
    out_dbn = L * ["."]
    for left, right, pair_set in zip(PSEUDO_LEFT, PSEUDO_RIGHT, pair_hierarch):
        for pair in pair_set:
            i, j = min(pair), max(pair)
            out_dbn[i] = left
            out_dbn[j] = right
    return "".join(out_dbn)


def mat2db(mat):
    mat_ = mat.squeeze()
    L = mat_.shape[0]
    ii, jj = torch.where(mat_)
    ii, jj = ii.tolist(), jj.tolist()
    idxs = list(zip(ii, jj))
    return dict2db(idxs, L)


def embed_bpseq(path, ex_noncanon=False, ex_sharp=False):
    BASES = dict(A=0, C=1, G=2, U=3)
    CANON = {"AU", "UA", "CG", "GC", "UG", "GU"}
    with open(path, "r") as f:
        lines = f.read().splitlines()
    seq = ""
    second = {}
    line_idx = 0
    con_idxs = []
    for line in lines:
        if line.startswith("#"):
            continue
        words = line.split()
        base = words[1]
        seq += base
    for line in lines:
        if line.startswith("#"):
            continue
        words = line.split()
        ii, jj = int(words[0])-1, int(words[-1])-1
        if ii < jj:
            pair = seq[ii] + seq[jj]
            if ex_noncanon and pair not in CANON:
                continue
            if ex_sharp and abs(ii - jj) < 4:
                continue
            con_idxs.append((ii, jj))
            con_idxs.append((jj, ii))
    fasta = torch.zeros(4, len(seq), dtype=torch.uint8)
    idxs = [BASES[char] for char in seq]
    fasta[idxs, list(range(len(seq)))] = 1
    dbn = dict2db(con_idxs, len(seq))
    con = torch.zeros(len(seq), len(seq), dtype=torch.uint8)
    if len(con_idxs) > 0:
        con_idxs_ = torch.tensor(con_idxs)# + offset
        con[con_idxs_[:,0], con_idxs_[:,1]] = 1
        con[con_idxs_[:,1], con_idxs_[:,0]] = 1
    return con, seq, dbn


def run_pet(fasta, suboptimal=0, ppfile=False, ppfold=False, return_dbn=False):
    # write fasta to output file
    tmp_fasta = "/tmp/pet.fasta"
    tmp_pfile = "/tmp/pet.txt"
    L = len(fasta)
    with open(tmp_fasta, "w") as f:
        f.write(f">tmp\n{fasta}\n")
    # build command
    command = [EXE, "--verbose", "-f", tmp_fasta]
    if ppfile:
        command.extend(["--ppfile", tmp_pfile])
    # if ppfold:
    #     command.append("--ppfold")
    if suboptimal > 0:
        command.extend(["--suboptimal", str(suboptimal)])
    output = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_list = []
    # rnafold output
    pattern = re.compile(r"Sequence  1 structure = (?P<db>[\|\.\,\(\)\{\}]{4,})", re.MULTILINE)
    match = pattern.search(output.stdout)
    if return_dbn:
        out_list.append((match.group("db"), parsedb(match.group("db"))))
    else:
        out_list.append(parsedb(match.group("db")))
    # pfold output
    pattern = re.compile(r"Pfold RNA structure:\t(?P<db>[\.\,\(\)\{\}]{4,})", re.MULTILINE)
    match = pattern.search(output.stdout)
    if return_dbn:
        out_list.append((match.group("db"), parsedb(match.group("db"))))
    else:
        out_list.append(parsedb(match.group("db")))
    # petfold output
    pattern = re.compile(r"PETfold RNA structure:\t(?P<db>[\.\,\(\)\{\}]{4,})", re.MULTILINE)
    match = pattern.search(output.stdout)
    if return_dbn:
        out_list.append((match.group("db"), parsedb(match.group("db"))))
    else:
        out_list.append(parsedb(match.group("db")))
    # pfile
    if ppfile:
        prob_mat = torch.zeros(L, L)
        with open("/tmp/pet.txt", "r") as f:
            lines = f.read().splitlines()
        for i, line in enumerate(lines[1:-2]):
            prob_mat[i, :] = torch.tensor(list(map(float, line.split())))
        out_list.append(prob_mat)
    # suboptimal structs
    if suboptimal > 0:
        pattern = re.compile(r"Suboptimal structure:   (?P<db>[\.\,\(\)\{\}]{4,})\t(?P<score>0\.\d+)", re.MULTILINE)
        match_itr = pattern.finditer(output.stdout)
        dbn_score = [(match.group("db"), float(match.group("score"))) for match in match_itr]
        dbn_score.sort(key=lambda tup: tup[1], reverse=True)
        subopts = [parsedb(dbn) for dbn, score in dbn_score]
        out_list.append(subopts)
    return out_list


def run_mxfold2(seq, return_mat=True):
    tmp_fasta = "/tmp/pet.fasta"
    with open(tmp_fasta, "w") as f: 
        f.write(f">tmp\n{seq}\n")
    # build command
    command = ["mxfold2", "predict", tmp_fasta]
    output = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    pattern = re.compile(r"(?P<db>[\|\.\,\(\)\{\}]{4,})", re.MULTILINE)
    match = pattern.search(output.stdout)
    dbn = match.group("db")
    if not return_mat:
        return dbn
    return dbn, parsedb(dbn)


def run_spotrna(seq):
    tmp_fasta = "/tmp/temp.fasta"
    tmp_bpseq = "/tmp/temp.bpseq"
    with open(tmp_fasta, "w") as f: 
        f.write(f">tmp\n{seq}\n")
    # build command
    command = ["python", SPOT_EXE, "--inputs", tmp_fasta, "--outputs", "/tmp/"]
    output = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return output


def plot_dbn(struct, seq, ax = None, size=1):
    if ax is None:
        fig, ax = plt.subplots()
    coord_objs = RNA.simple_xy_coordinates(struct)
    coords_tups = [(coord.X, coord.Y) for coord in coord_objs[:-1]]
    coords = np.array(coords_tups)
    coords -= coords.min(0)
    coords /= coords.max(0)
    for char, (x, y) in zip(seq, coords):
        ax.text(x, y,
                size=size,
                ha="center",
                va="center",
                s=char,
                bbox=dict(boxstyle="circle",
                          color=COLORS[char]))
    nest_idxs, pseudo_idxs = parsedb(struct, return_idxs=True)
    for i, j in nest_idxs:
        if i < j:
            con = patches.ConnectionPatch(coords[i], coords[j], ax.transData, linewidth=1, color="black")
            ax.add_artist(con)
    for i, j in pseudo_idxs:
        if i < j:
            con = patches.ConnectionPatch(coords[i], coords[j], ax.transData, linewidth=2, color="red")
            ax.add_artist(con)
    for first, second in zip(coords[:-1,:], coords[1:,:]):
        con = patches.ConnectionPatch(first, second, ax.transData, linewidth=1, color="gray")
        ax.add_artist(con)
    ax.set(aspect=1.0)
    ax.axis("off")
    return ax


def run_prob_pair(seq):
    tmp_fasta = "/tmp/prob.fasta"
    tmp_ct = "/tmp/prob.ct"
    with open(tmp_fasta, "w") as f: 
        f.write(f">tmp\n{seq}\n")
    # build command
    command = [PROB_PAIRS_EXE, "--sequence", tmp_fasta, tmp_ct]
    res = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    ct_mat = torch.from_numpy(np.loadtxt(tmp_ct))
    out = torch.zeros(len(seq), len(seq))
    if len(ct_mat) > 0:
        ii = ct_mat[:,0].long() - 1
        jj = ct_mat[:,1].long() - 1
        probs = ct_mat[:,-1].float()
        out[ii, jj] = out[jj, ii] = probs

    return out
