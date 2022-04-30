import sys
import json
from kirigami.core import train
from kirigami.nn import loss
import torch
import torch.nn as nn


def main():
    if len(sys.argv) == 2:
        out_file = sys.argv[1] + ".pt"
    else:
        out_file = None

    dist_idxs = [0]
    # dist_idxs = list(range(10)) 
    bins = "torch.arange(2,21,.5)"

    train(notes=None,
          data=dict(tr_set="scripts/TR1234.pt",
                    vl_set="scripts/VL128.pt",
                    ceiling=80, 
                    multiclass=True,
                    bins=bins,
                    use_thermo=True,
                    use_dist=True,
                    inv=False,
                    dist_idxs=dist_idxs,
                    inv_eps=1e-8,
                    batch_size=1,
                    batch_sample=True),
          criterion=("loss.ForkLoss(dist_crit=loss.CEMulti(),"
                                   "pos_weight=.90,"
                                   "inv_weight=.00,"
                                   "bin_weight=.10,"
                                   "dropout=False)"),
          resume=False,
          eval_freq=1,
          bar=False,
          out_file=out_file,
          save_epoch=1000,
          tr_chk=None,
          vl_chk=None,
          device="cuda",
          layers=["kirigami.nn.ResNet(in_channels=9,act='ReLU',n_blocks=4,p=0.2,n_channels=32)",
                  f"kirigami.nn.Fork(in_channels=32,out_dists={len(dist_idxs)},multiclass=True,n_bins={len(eval(bins))},kernel_size=3)",
                  "kirigami.nn.Symmetrize()"],
          optimizer="torch.optim.Adam",
          lr=0.001,
          epochs=2000, 
          iter_acc=1,
          mix_prec=False,
          chkpt_seg=4,
          symmetrize=True,
          canonicalize=True,
          thres_by_ground_pairs=True,
          thres_prob=0.0)

if __name__ == "__main__":
    main()
