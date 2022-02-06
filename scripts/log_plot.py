import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec
import torch

def main():
    with open(sys.argv[1], "r") as f:
        lines = f.read().splitlines()
    
    tot_train_loss = []
    dist_loss = []
    con_loss = []

    val_dist_loss = []
    val_con_loss = []
    mean_mcc_ = []

    L1, L2, L5, L10, tot_dist = [], [], [], [], []
    L1_PCC, L2_PCC, L5_PCC, L10_PCC, tot_PCC = [], [], [], [], []

    n_epochs = 0
    eval_freq = 1
    for line in lines:
        if line.startswith("Beginning"):
            n_epochs += 1
            continue
        elif "eval_freq" in line:
            last_word = line.split()[1]
            last_word = last_word.replace(",", "")
            eval_freq = int(last_word)
        try:
            val = float(line.split(": ")[1])
        except:
            continue
        if line.startswith("Mean total training loss"): 
            tot_train_loss.append(val)
        elif line.startswith("Mean distance loss"): 
            dist_loss.append(val)
        elif line.startswith("Mean contact loss"):
            con_loss.append(val)
        elif line.startswith("Raw mean validation"):
            val_loss.append(val)
        elif line.startswith("Mean contact validation loss"):
            val_con_loss.append(val)
        elif line.startswith("Mean distance validation loss"):
            val_dist_loss.append(val)
        elif line.startswith("Mean MCC"):
            mean_mcc_.append(val)
        elif "total distance error" in line:
            tot_dist.append(val)
        elif "error (first L)" in line:
            L1.append(val)
        elif "error (first 2L)" in line:
            L2.append(val)
        elif "error (first 5L)" in line:
            L5.append(val)
        elif "error (first 10L)" in line:
            L10.append(val)
        elif "total distance PCC" in line:
            L1_PCC.append(val)
        elif "PCC (first L)" in line:
            L2_PCC.append(val)
        elif "PCC (first 2L)" in line:
            L5_PCC.append(val)
        elif "PCC (first 5L)" in line:
            L10_PCC.append(val)
        elif "PCC (first 10L)" in line:
            tot_PCC.append(val)

    mean_mcc = np.array(mean_mcc_)
    max_mcc = mean_mcc.max()
    max_mcc_epoch = eval_freq * mean_mcc.argmax()

    fig = plt.figure(figsize=(10,10))
    gs = gridspec.GridSpec(nrows=3, ncols=2)
    # fig, axs = plt.subplots(3, 2, constrained_layout=True, figsize=(10,10))
    # axs = axs.ravel()

    if len(sys.argv) == 3:
        plt.suptitle(sys.argv[2])
            
    ax0 = fig.add_subplot(gs[0,0])
    ax0.set_title("VL MCC")
    ax0.plot(eval_freq * np.arange(len(mean_mcc)), mean_mcc)
    ax0.axvline(max_mcc_epoch, color="red", linestyle="--")
    ax0.axhline(max_mcc, color="red", linestyle="--",
                   label=f"Max = {max_mcc:.5} at epoch {max_mcc_epoch}")
    ax0.set_xlabel("Epoch")
    ax0.set_ylabel("MCC")
    ax0.set_yticks(np.arange(0, 1.1, .1))
    ax0.minorticks_on()
    ax0.grid(True)
    ax0.grid(which="major", linewidth=1)
    ax0.grid(which="minor", linewidth=0.5)
    ax0.legend()

    x = eval_freq * np.arange(len(val_dist_loss))
    ax1 = fig.add_subplot(gs[0,1])
    ax1.set_title("Distance Loss")
    ax1.plot(dist_loss, label="TR")
    ax1.plot(x, val_dist_loss, label="VL")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.legend()

    x = eval_freq * np.arange(len(tot_dist))
    ax2 = fig.add_subplot(gs[1,0])
    ax2.set_title("VL Distance L1 Error")
    ax2.plot(x, tot_dist, label="Total")
    ax2.plot(x, L1, label="L")
    ax2.plot(x, L2, label="2L")
    ax2.plot(x, L5, label="5L")
    ax2.plot(x, L10, label="10L")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Error")
    # axs[2].set_ylim((0, 20))
    ax2.grid(True)
    ax2.minorticks_on()
    ax2.grid(which="major", linewidth=1)
    ax2.grid(which="minor", linewidth=.5)
    ax2.legend()

    x = eval_freq * np.arange(len(val_con_loss))
    ax3 = fig.add_subplot(gs[1,1])
    ax3.set_title("Contact Loss")
    ax3.plot(con_loss, label="TR")
    ax3.plot(x, val_con_loss, label="VL")
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.legend()

    x = eval_freq * np.arange(len(tot_PCC))
    ax4 = fig.add_subplot(gs[2,:])
    ax4.set_title("VL Distance PCC")
    ax4.plot(x, tot_PCC, label="Total")
    ax4.plot(x, L1_PCC, label="L")
    ax4.plot(x, L2_PCC, label="2L")
    ax4.plot(x, L5_PCC, label="5L")
    ax4.plot(x, L10_PCC, label="10L")
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("PCC")
    ax4.set_ylim((-1, 1))
    ax4.grid(True)
    ax4.minorticks_on()
    ax4.grid(which="major", linewidth=1)
    ax4.grid(which="minor", linewidth=.5)
    ax4.legend()

    plt.tight_layout()

    base_name = os.path.basename(sys.argv[1])
    base_name, _ = os.path.splitext(base_name)
    png_name = base_name + ".png"
    plt.savefig(png_name, dpi=300)


if __name__ == "__main__":
    main()
