import sys
import json
from kirigami.core import train
import torch

def main():
    if len(sys.argv) == 2:
        out_file = sys.argv[1] + ".pt"
    else:
        out_file = None
    
    train(notes="",
          resume=False,
          eval_freq=10,
          bar=False,
          use_thermo=True,
          use_dist=True,
          n_dists=10,
          dist_kwargs=dict(ceiling=50, bins=torch.arange(0, 50, 5)),
          tr_set="scripts/TR1_norm_clip_50.pt",
          vl_set="scripts/VL1_norm_clip_50.pt",
          out_file=out_file,
          save_epoch=1990,
          batch_size=1,
          batch_sample=True,
          tr_chk=None,
          vl_chk=None,
          device="cuda",
          layers=["kirigami.nn.ResNet(in_channels=9, n_blocks=8, norm='InstanceNorm2d', n_channels=32)",
                  "kirigami.nn.Fork(in_channels=1, out_dist_channels=10, kernel_size=3)",
                  "kirigami.nn.Symmetrize()"],
          criterion="ForkL1(pos_weight=0.6, dist_weight=0.5, dropout=False, inv=False, f=None)",
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
