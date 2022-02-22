import torch
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
from tqdm import tqdm


def main():
    data_path = sys.argv[1]
    data = torch.load(data_path)

    out_path = data_path.split(".")[0]
    os.mkdir(out_path)
    
    prd_dists_ = []
    grd_dists_ = []
    for i, row in enumerate(tqdm(data)):
        (grd_con, grd_dist), (prd_con, prd_dist) = row
        fig, axs = plt.subplots(3, 2, figsize=(10,10), constrained_layout=True)

        grd_dist *= 50
        prd_dist *= 50 
        grd_dist = grd_dist.clip(0, 23)
        prd_dist = prd_dist.clip(0, 23)
        
        axs[0,0].imshow(grd_con.cpu().numpy().squeeze())
        axs[0,0].set_title("Ground Contact")
        axs[0,1].imshow(prd_con.cpu().numpy().squeeze())
        axs[0,1].set_title("Predicted Contact")
        axs[1,0].imshow(grd_dist[:,0,:,:].cpu().numpy().squeeze(), vmin=0, vmax=23)
        axs[1,0].set_title("Ground P-P")
        axs[1,1].imshow(prd_dist[:,0,:,:].cpu().numpy().squeeze(), vmin=0, vmax=23)
        axs[1,1].set_title("Predicted P-P")
        axs[2,0].imshow(grd_dist[:,-1,:,:].cpu().numpy().squeeze(), vmin=0, vmax=23)
        axs[2,0].set_title("Ground truth O-O")
        axs[2,1].imshow(prd_dist[:,-1,:,:].cpu().numpy().squeeze(), vmin=0, vmax=23)
        axs[2,1].set_title("Predicted O-O")
        
        plt.savefig(f"{out_path}/mol{i:02}.png")
        plt.close()

        prd_dists_.append(prd_dist.flatten())
        grd_dists_.append(grd_dist.flatten())

    prd_dists = torch.cat(prd_dists_, 0)
    grd_dists = torch.cat(grd_dists_, 0)

    # prd_dists = prd_dists[prd_dists > 0]
    # grd_dists = grd_dists[grd_dists > 0]
    # grd_sort_obj = torch.sort(grd_dists)

    # grd_sort = 23 * grd_sort_obj.values
    # prd_sort = 23 * prd_dists[grd_sort_obj.indices]

    # grd_sort = grd_sort[::100].cpu().numpy()
    # prd_sort = prd_sort[::100].cpu().numpy()

    # grd_sort = grd_sort[grd_sort < 10]
    # prd_sort = prd_sort[:len(grd_sort)]

    grd_sort = grd_dists.cpu().numpy()
    prd_sort = prd_dists.cpu().numpy()
    

    fig, axs = plt.subplots(2, 1, figsize=(20,20))
    axs[0].scatter(grd_sort, prd_sort, marker=".", s=1)
    axs[0].set_xlabel("Ground truth")
    axs[0].set_ylabel("Predicted")
    axs[0].set_xlim((0, 23))
    axs[0].set_ylim((0, 23))
    # axs[0].plot(np.arange(5,23), np.arange(5,23), color="r", linestyle="--", alpha=.5, label="y=x")
    axs[0].plot(np.arange(5,23), np.arange(5,23), color="r", linestyle="--", alpha=.5, label="y=x")
    axs[1].hist(grd_sort, bins=100, alpha=.5, label="Ground truth")
    axs[1].hist(prd_sort, bins=100, alpha=.5, label="Predicted")
    axs[1].set_ylim((0,50_000))
    plt.legend()
    # plt.plot(np.arange(5,10), np.arange(5,10), color="r", linestyle="--", alpha=.5, label="y=x")
    # plt.savefig(f"{out_path}/out.png", dpi=500) 
    plt.savefig(f"{out_path}_dist.png", dpi=300) 

    


if __name__ == "__main__":
    main()
