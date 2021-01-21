import json
import os
import pathlib
import munch


def path2munch(path: pathlib.Path) -> munch.Munch:
    with open(path, 'r') as f:
        txt = f.read()
    conf_dict = json.loads(txt)
    return munch.munchify(conf_dict)

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

