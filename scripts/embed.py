"""
Embeds tensors
"""

import os
from tqdm import tqdm
from glob import glob
import torch
from kirigami import *
from kirigami.containers.molecule import Molecule
from torch.utils.data import TensorDataset

def main():
    FULL_LENGTH = 512
    # FOLDER = "/gpfs/ysm/project/pyle/mah258/spot/bpRNA/TR0-bpseq-cleaned"
    # FOLDER = "/gpfs/ysm/project/pyle/mah258/spot/bpRNA/VL0-bpseq-cleaned"

    files = glob(os.path.join(FOLDER, "*"))
    files.sort()

    _contacts = []
    _fastas = []

    for file in tqdm(files):
        mol = Molecule.from_file(file)

        fasta_len = mol.fasta.shape[1]
        offset = (FULL_LENGTH - fasta_len) // 2
        fasta = torch.zeros(4, FULL_LENGTH, dtype=torch.uint8)
        fasta[:, offset:offset+fasta_len] = mol.fasta
        _fastas.append(fasta.to_sparse())

        _contact = mol.contact
        contact = torch.zeros(FULL_LENGTH, FULL_LENGTH, dtype=torch.uint8)
        contact[offset:offset+fasta_len, offset:offset+fasta_len] = _contact
        _contacts.append(contact.to_sparse())

    contacts = torch.stack(_contacts)
    fastas = torch.stack(_fastas)
    dset = TensorDataset(contacts, fastas)
    torch.save(dset, "VL0.pt")


    # print(out_fasta.shape)
    # out = mol.to_tensor()
    # tensor_list = []
    # for file in tqdm(self.in_files):
    #     wrapper = self.wrapper.from_file(file)
    #     tensor_list.append(wrapper.to_tensor(**self.kwargs))
    # tensor_list = list(zip(*tensor_list))
    # tensor_stacks = []
    # for _ in range(len(tensor_list)):
    #     tensors = tensor_list.pop()
    #     tensor_stack = torch.stack(tensors)
    #     del tensors
    #     tensor_stacks.append(tensor_stack)
    #     del tensor_stack
    # dset = torch.utils.data.TensorDataset(*tensor_stacks)
    # torch.save(dset, self.out_file)
    #   pass

if __name__ == "__main__":
    main()
