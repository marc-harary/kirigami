import torch

def main():
    dset = torch.load("VL1_pdb.pt")
    print(len(dset))

if __name__ == "__main__":
    main()
