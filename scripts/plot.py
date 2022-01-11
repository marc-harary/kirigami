import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec
import torch

def main():
    # with open(sys.argv[1], "r") as f:
    #     lines = f.read().splitlines()

    # train_loss = []
    # val_loss = []
    # mean_mcc = []
    # bin_loss = []
    # n_epochs = 0
    # eval_freq = 1
    # for line in lines:
    #     if line.startswith("Mean training loss"): 
    #         val = float(line.split(": ")[1])
    #         train_loss.append(val)
    #     elif line.startswith("Raw mean validation"):
    #         val = float(line.split(": ")[1])
    #         val_loss.append(val)
    #     elif line.startswith("Mean MCC"):
    #         val = float(line.split(": ")[1])
    #         mean_mcc.append(val)
    #     elif line.startswith("Binarized"):
    #         val = float(line.split(": ")[1])
    #         bin_loss.append(val)
    #     elif line.startswith("Beginning"):
    #         n_epochs += 1
    #     elif "eval_freq" in line:
    #         last_word = line.split()[1]
    #         last_word = last_word.replace(",", "")
    #         eval_freq = int(last_word)

    data = torch.load(sys.argv[1])

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(list(data["mcc_history"].keys()), list(data["mcc_history"].values()))
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MCC")
    ax1.set_title("MCC by Epoch")
    ax1.set_yticks(np.arange(0, 1, 0.1))

    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(list(data["vl_loss_history"].keys()), list(data["vl_loss_history"].values()))
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Loss")
    ax2.set_title("VL Loss by Epoch")

    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(list(data["tr_loss_history"].keys()), list(data["tr_loss_history"].values()))
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Raw VL Loss by Epoch")

    # plt.tight_layout()
    
    base_name = os.path.basename(sys.argv[1])
    base_name, _ = os.path.splitext(base_name)
    png_name = base_name + ".png"
    plt.savefig(png_name, dpi=300)

    # npy_name = base_name + "_mcc.npy"
    # np.save(npy_name, mean_mcc)
        
         

if __name__ == "__main__":
    main()
