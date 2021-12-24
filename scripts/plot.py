import sys
import matplotlib.pyplot as plt
import numpy as np
import os

def main():
    with open(sys.argv[1], "r") as f:
        lines = f.read().splitlines()

    train_loss = []
    val_loss = []
    mean_mcc = []
    bin_loss = []
    n_epochs = 0
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

    plt.subplot(221).plot(range(len(mean_mcc)), mean_mcc)
    plt.gca().set_xlabel("Epoch")
    plt.gca().set_ylabel("MCC")
    plt.gca().set_title("MCC by Epoch")

    plt.subplot(222).plot(range(len(train_loss)), train_loss)
    plt.gca().set_xlabel("Epoch")
    plt.gca().set_ylabel("Loss")
    plt.gca().set_title("TR Loss by Epoch")

    plt.subplot(223).plot(range(len(val_loss)), val_loss)
    plt.gca().set_xlabel("Epoch")
    plt.gca().set_ylabel("Loss")
    plt.gca().set_title("Raw VL Loss by Epoch")

    plt.subplot(224).plot(range(len(bin_loss)), bin_loss)
    plt.gca().set_xlabel("Epoch")
    plt.gca().set_ylabel("Loss")
    plt.gca().set_title("Binarized VL Loss by Epoch")

    plt.tight_layout()
    
    base_name = os.path.basename(sys.argv[1])
    base_name, _ = os.path.splitext(base_name)
    base_name += ".png"
    plt.savefig(base_name)
        
         

if __name__ == "__main__":
    main()
