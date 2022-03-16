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

    dist_idxs = [0, 1, 2]
    bins = torch.arange(2, 20.5, .5)
    bins = torch.cat([torch.tensor([0]), bins])
    
    train(notes="",
          data=dict(tr_set="scripts/TR1.pt",
                    vl_set="scripts/VL1.pt",
                    # vl_set=None,
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
          # crit_kwargs=dict(pos_weight=.6,
          #                  dist_weight=.5,
          #                  dist_crit=nn.L1Loss(reduction="none")),
          # criterion=loss.ForkLoss(dist_crit=nn.L1Loss(reduction="none"),
          criterion=loss.ForkLoss(dist_crit=loss.CEMulti(),
                                  pos_weight=.6,
                                  dist_weight=.9,
                                  dropout=False),
          resume=False,
          eval_freq=10,
          bar=False,
          out_file=out_file,
          save_epoch=1990,
          tr_chk=None,
          vl_chk=None,
          device="cuda",
          layers=["kirigami.nn.ResNet(in_channels=9, n_blocks=8, norm='InstanceNorm2d', n_channels=32)",
                  f"kirigami.nn.Fork(in_channels=1, out_dists={len(dist_idxs)}, multiclass=True, n_bins={len(bins)}, kernel_size=3)",
                  "kirigami.nn.Symmetrize()"],
          optimizer="torch.optim.Adam",
          lr=0.001,
          epochs=2000,
          iter_acc=1,
          mix_prec=False,
          chkpt_seg=1,
          symmetrize=True,
          canonicalize=True,
          thres_by_ground_pairs=True,
          thres_prob=0.0)

if __name__ == "__main__":
    main()
