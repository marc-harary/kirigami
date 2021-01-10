from collections import deque
import re
import torch
from multipledispatch import dispatch
from .constants import *

@dispatch(str)
def embed_fasta(seq_str):   
    seq_list = torch.stack([BASE_DICT[char] for char in seq_str.lower()])
    return torch.unsqueeze(seq_list, 0)

@dispatch(list)
def embed_fasta(seq_list):
    return list(map(embed_fasta, seq_list))

@dispatch(str)
def embed_bpseq(bpseq_str):
    lines = bpseq_str.splitlines()
    idx_str = [int(line.split()[-1]) for line in lines]
    idx_ten = torch.Tensor(idx_str).to(torch.int64)
    ret = F.one_hot(idx_ten, num_classes=len(idx_ten)+1)
    return ret[:,1:]

@dispatch(list)
def embed_bpseq(bpseq_list):
    return list(map(embed_bpseq, bpseq_list))

def read_label(infile):
    label_list=[]
    fp=open(infile,'r')
    lines=fp.read().splitlines()
    fp.close()
    for line in lines:
        if line.startswith('#') or line.startswith('i'):
            continue
        items=line.split()
        nt1=int(items[0])
        nt2=int(items[1])
        if nt1 < nt2:
            label_list.append((nt1,nt2))
        else:
            label_list.append((nt2,nt1))
    BPnat=len(label_list)
    return label_list, BPnat

def calcF1MCC(sequence,positive_list,predict_list):
    L=len(sequence)
    total=L*(L-1)/2
    predicted=set(predict_list)
    positive=set(positive_list)
    TP=1.*len(predicted.intersection(positive))
    FP=len(predicted)-TP
    FN=len(positive)-TP
    TN=total-TP-FP-FN
    if len(predicted)==0 or len(positive)==0:
        return 0,0
    precision=TP/len(predicted)
    recall=TP/len(positive)
    F1=0
    MCC=0
    if TP>0:
        F1=2/(1/precision+1/recall)
    if (TP+FP)*(TP+FN)*(TN+FP)*(TN+FN)>0:
        MCC=(TP*TN-FP*FN)/((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**.5
    return F1,MCC

