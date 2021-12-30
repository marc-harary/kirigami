import sys
import json
from kirigami.core import train

def main():
    train(notes="Run with batch sampling with .sum() rather than .mean()",
          resume=False,
          eval_freq=5,
          bar=False,
          thermo=False,
          tr_set="/home/mah258/kirigami/data/TR0-list.pt",
          vl_set="/home/mah258/kirigami/data/VL0-list.pt",
          batch_size=1,
          batch_sample=True,
          tr_chk="/home/mah258/kirigami/exps/tr_chkpt.pt",
          vl_chk="/home/mah258/kirigami/exps/vl_chkpt.pt",
          device="cuda",
          layers=["kirigami.nn.ResNet(n_blocks=1, n_channels=16)"],
          criterion="WeightLoss(weight=.6)",
          optimizer="torch.optim.Adam",
          lr=0.001,
          epochs=1000,
          iter_acc=1,
          mix_prec=False,
          chkpt_seg=4,
          symmetrize=True,
          canonicalize=True,
          thres_by_ground_pairs=True,
          thres_prob=0.0)

if __name__ == "__main__":
    main()
