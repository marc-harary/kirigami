import sys
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.gridspec as gridspec

def main():
    with open(sys.argv[1], "r") as f:
        lines = f.read().splitlines()

    train_loss = []
    val_loss = []
    mean_mcc = []
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
            mean_mcc.append(val)
        elif line.startswith("Binarized"):
            val = float(line.split(": ")[1])
            bin_loss.append(val)
        elif line.startswith("Beginning"):
            n_epochs += 1
        elif "eval_freq" in line:
            last_word = line.split()[1]
            last_word = last_word.replace(",", "")
            eval_freq = int(last_word)

    fig = plt.figure(tight_layout=True)
    gs = gridspec.GridSpec(2, 2)

    ax1 = fig.add_subplot(gs[0,:])
    ax1.plot(range(len(train_loss)), train_loss)
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title("TR Loss by Epoch")

    ax2 = fig.add_subplot(gs[1,0])
    ax2.plot(eval_freq*np.arange(len(val_loss)), mean_mcc)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("MCC")
    ax2.set_title("MCC by Epoch")

    ax3 = fig.add_subplot(gs[1,1])
    ax3.plot(eval_freq*np.arange(len(val_loss)), val_loss)
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Loss")
    ax3.set_title("Raw VL Loss by Epoch")

    # plt.subplot(224).plot(range(len(bin_loss)), bin_loss)
    # plt.gca().set_xlabel("Epoch")
    # plt.gca().set_ylabel("Loss")
    # plt.gca().set_title("Binarized VL Loss by Epoch")

    # plt.tight_layout()
    
    base_name = os.path.basename(sys.argv[1])
    base_name, _ = os.path.splitext(base_name)
    base_name += ".png"
    plt.savefig(base_name)
        
         

if __name__ == "__main__":
    main()
