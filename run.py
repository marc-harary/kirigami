import sys
import json
from kirigami.core import train

def main():
    train(notes="N/A",
          resume=False,
          eval_freq=1,
          bar=False,
          thermo=True,
          tr_set="/gpfs/ysm/project/pyle/mah258/data/TR1.pt",
          vl_set="/gpfs/ysm/project/pyle/mah258/data/VL1.pt",
          batch_size=1,
          batch_sample=True,
          # tr_chk="/home/mah258/kirigami/tr_chk.pt",
          # vl_chk="/home/mah258/kirigami/vl_chk.pt",
          tr_chk=None,
          vl_chk=None,
          device="cuda",
          layers=["kirigami.nn.ResNet(in_channels=9, n_blocks=16, norm='BatchNorm2d', n_channels=8)"],
          criterion="WeightLoss(weight=.6)",
          optimizer="torch.optim.Adam",
          lr=0.001,
          epochs=2000,
          iter_acc=8,
          mix_prec=False,
          chkpt_seg=1,
          symmetrize=True,
          canonicalize=True,
          thres_by_ground_pairs=True,
          thres_prob=0.0)

if __name__ == "__main__":
    main()
