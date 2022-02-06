import sys
import json
from kirigami.core import train

def main():
    train(notes="L1 non-inverted loss; clipped to 23; prob dropout",
          resume=False,
          eval_freq=5,
          bar=False,
          use_thermo=True,
          use_dist=True,
          n_dists=10,
          dist_kwargs=dict(ceiling=23),
          tr_set="scripts/TR1_norm_clip_23.pt",
          vl_set="scripts/VL1_norm_clip_23.pt",
          batch_size=1,
          batch_sample=True,
          # tr_chk="/home/mah258/kirigami/tr_chk.pt",
          # vl_chk="/home/mah258/kirigami/vl_chk.pt",
          tr_chk=None,
          vl_chk=None,
          device="cuda",
          layers=["kirigami.nn.ResNet(in_channels=9, n_blocks=8, norm='InstanceNorm2d', n_channels=32)",
                  "kirigami.nn.Fork(in_channels=1, out_dist_channels=10, kernel_size=3)",
                  "kirigami.nn.Symmetrize()"],
          criterion="ForkL1(pos_weight=.6, dist_weight=0.5)",
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
