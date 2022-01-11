import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec
import torch

def main():
    with open(sys.argv[1], "r") as f:
        lines = f.read().splitlines()
    
    train_loss = []
    val_loss = []
    mean_mcc_ = []
    bin_loss = []
    n_epochs = 0
    eval_freq = 1
    for line in lines:
        if line.startswith("Mean training loss"): 
            val = float(line.split(": ")[1])
            train_loss.append(val)
        elif line.startswith("Raw mean validation"):
            val = float(line.split(": ")[1])
            val_loss.append(val)
        elif line.startswith("Mean MCC"):
            val = float(line.split(": ")[1])
            mean_mcc_.append(val)
        elif line.startswith("Binarized"):
            val = float(line.split(": ")[1])
            bin_loss.append(val)
        elif line.startswith("Beginning"):
            n_epochs += 1
        elif "eval_freq" in line:
            last_word = line.split()[1]
            last_word = last_word.replace(",", "")
            eval_freq = int(last_word)

    mean_mcc = np.array(mean_mcc_)
    max_mcc = mean_mcc.max()
    max_mcc_epoch = mean_mcc.argmax()

    fig, (ax1, ax2) = plt.subplots(2, 1, constrained_layout=True)

    if len(sys.argv) == 3:
        plt.suptitle(sys.argv[2])
            
    ax1.plot(mean_mcc)
    ax1.axvline(max_mcc_epoch, color="red", linestyle="--")
    ax1.axhline(max_mcc, color="red", linestyle="--",
                label=f"Max = {max_mcc:.5} at epoch {max_mcc_epoch}")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("MCC")
    ax1.set_title("MCC by Epoch")
    ax1.set_yticks(np.arange(0, 1.1, .1))
    ax1.minorticks_on()
    ax1.grid(True)
    ax1.grid(which="major", linewidth=1)
    ax1.grid(which="minor", linewidth=0.5)
    ax1.legend()

    ax2.plot(train_loss, label="Training loss")
    ax2.plot(val_loss, label="Validation loss")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Log loss")
    ax2.set_title("Log Loss by Epoch")
    ax2.set_yscale("log")
    ax2.legend()

    base_name = os.path.basename(sys.argv[1])
    base_name, _ = os.path.splitext(base_name)
    png_name = base_name + ".png"
    plt.savefig(png_name, dpi=300)


if __name__ == "__main__":
    main()
