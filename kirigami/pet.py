import os
import subprocess
import re
from collections import deque
import numpy as np
import torch


ROOT = os.path.dirname(__file__)
EXE = os.path.join(ROOT, "PETfold")
os.environ["PETFOLDBIN"] = ROOT


def parse_db(db_str):
    # db_str = db_str.replace("{", "(")
    # db_str = db_str.replace("}", ")")
    # db_str = db_str.replace(",", ".")
    # db_str = db_str.replace("|", ".")
    # db_str = db_str.replace("[", "(")
    # db_str = db_str.replace(")", "]")
    L = len(db_str)
    stack = deque()
    # pair_dict[j] = i
    # out = np.zeros((L, L))
    out = torch.zeros(L, L)
    for i, char in enumerate(db_str):
        if char == "(":
            stack.append(i)
        elif char == ")":
            try:
                j = stack.pop()
            except:
                continue
            # pair_dict[j] = i
            out[i, j] = out[j, i] = 1
    return out
             

# def run_pet(fasta, suboptimal=0, ppfile=False, ppfold=False):
#     # write fasta to output file
#     tmp_fasta = "/tmp/pet.fasta"
#     tmp_pfile = "/tmp/pet.txt"
#     L = len(fasta)
#     with open(tmp_fasta, "w") as f:
#         f.write(f">tmp\n{fasta}\n")
#     # build command
#     command = [EXE, "--verbose", "-f", tmp_fasta]
#     if ppfile:
#         command.extend(["--ppfile", tmp_pfile])
#     if ppfold:
#         command.append("--ppfold")
#     if suboptimal > 0:
#         command.extend(["--suboptimal", str(suboptimal)])
#     output = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#     out_list = []
#     # rnafold output
#     pattern = re.compile(r"Sequence  1 structure = (?P<db>[\|\.\,\(\)\{\}]{4,})", re.MULTILINE)
#     match = pattern.search(output.stdout)
#     out_list.append(parse_db(match.group("db")))
#     # pfold output
#     pattern = re.compile(r"Pfold RNA structure:\t(?P<db>[\.\,\(\)\{\}]{4,})", re.MULTILINE)
#     match = pattern.search(output.stdout)
#     out_list.append(parse_db(match.group("db")))
#     # petfold output
#     pattern = re.compile(r"PETfold RNA structure:\t(?P<db>[\.\,\(\)\{\}]{4,})", re.MULTILINE)
#     match = pattern.search(output.stdout)
#     out_list.append(parse_db(match.group("db")))
#     # pfile
#     if ppfile:
#         prob_mat = torch.zeros(L, L)
#         with open("/tmp/pet.txt", "r") as f:
#             lines = f.read().splitlines()
#         for i, line in enumerate(lines[1:-2]):
#             prob_mat[i, :] = torch.tensor(list(map(float, line.split())))
#         out_list.append(prob_mat)
#     # suboptimal structs
#     if suboptimal > 0:
#         pattern = re.compile(r"Suboptimal structure:   (?P<db>[\.\,\(\)\{\}]{4,})\t(?P<score>0\.\d+)", re.MULTILINE)
#         match_itr = pattern.finditer(output.stdout)
#         dbn_score = [(match.group("db"), float(match.group("score"))) for match in match_itr]
#         dbn_score.sort(key=lambda tup: tup[1], reverse=True)
#         subopts = [parse_db(dbn) for dbn, score in dbn_score]
#         out_list.append(subopts)
#     return out_list


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
    if ppfold:
        command.append("--ppfold")
    if suboptimal > 0:
        command.extend(["--suboptimal", str(suboptimal)])
    output = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    out_list = []
    # rnafold output
    pattern = re.compile(r"Sequence  1 structure = (?P<db>[\|\.\,\(\)\{\}]{4,})", re.MULTILINE)
    match = pattern.search(output.stdout)
    if return_dbn:
        out_list.append((match.group("db"), parse_db(match.group("db"))))
    else:
        out_list.append(parse_db(match.group("db")))
    # pfold output
    pattern = re.compile(r"Pfold RNA structure:\t(?P<db>[\.\,\(\)\{\}]{4,})", re.MULTILINE)
    match = pattern.search(output.stdout)
    if return_dbn:
        out_list.append((match.group("db"), parse_db(match.group("db"))))
    else:
        out_list.append(parse_db(match.group("db")))
    # petfold output
    pattern = re.compile(r"PETfold RNA structure:\t(?P<db>[\.\,\(\)\{\}]{4,})", re.MULTILINE)
    match = pattern.search(output.stdout)
    if return_dbn:
        out_list.append((match.group("db"), parse_db(match.group("db"))))
    else:
        out_list.append(parse_db(match.group("db")))
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
        subopts = [parse_db(dbn) for dbn, score in dbn_score]
        out_list.append(subopts)
    return out_list
