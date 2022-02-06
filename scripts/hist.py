import torch
import matplotlib.pyplot as plt
import numpy as np

def main():
    dset = torch.load("TR1.pt")
    dset = [tup[-1] for tup in dset]
    dists = torch.hstack([dist.flatten() for dist in dset])
    dists = dists[dists > 0].numpy()
    hist, bin_edges = np.histogram(dists, bins=100)
    idxs = np.digitize([4, 100], bin_edges)
    print(idxs)
    print(bin_edges[idxs])
    # plt.hist(dists, bins=100)
    # plt.savefig("hist_reg-4.png")

if __name__ == "__main__":
    main()
