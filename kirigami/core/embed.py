import argparse
import os
from glob import glob
from typing import *
from pathlib import Path
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
# from kirigami.utils.data import *
# from kirigami.utils.convert import *
from kirigami.utils import utils
# from kirigami.containers.distance import Distance
from kirigami.containers.contact import Contact
# from kirigami.containers.contact import Zuker


class Embed:
    """namespace for file embedding"""

    in_files: List[Path]
    out_file: Path
    # wrapper: Union[Distance, Contact]
    kwargs: Dict[str, Any]
    
    def __init__(self,
                 in_files: List[Path],
                 out_file: Path,
                 # wrapper: Union[Distance, Contact, Zuker],
                 wrapper: Union[Contact],
                 # dataset: Dataset,
                 **kwargs) -> None:
        self.in_files = in_files
        self.out_file = out_file
        self.wrapper = wrapper
        # self.dataset = dataset
        self.kwargs = kwargs
    
    def run(self) -> None:
        tensor_list = []
        for file in tqdm(self.in_files):
            wrapper = self.wrapper.from_file(file)
            tensor_list.append(wrapper.to_tensor(**self.kwargs))
        tensor_list = list(zip(*tensor_list))
        tensor_stacks = []
        for _ in range(len(tensor_list)):
            tensors = tensor_list.pop()
            tensor_stack = torch.stack(tensors)
            del tensors
            tensor_stacks.append(tensor_stack)
            del tensor_stack
        dset = torch.utils.data.TensorDataset(*tensor_stacks)
        torch.save(dset, self.out_file)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "Embed":
        if args.in_directory is not None:
            in_files = glob(os.path.join(args.in_directory, "*"))
        elif hasattr(args, "in_list"):
            with open(args.in_list, "r") as f:
                in_files = f.read().splitlines()
                in_files = list(map(lambda string: string.strip(), in_files))
        if args.file_type == "contact":
            return cls(in_files=in_files,
                       out_file=args.out_file,
                       wrapper=Contact,
                       # dataset=ContactDataset,
                       dim=args.dim,
                       length=args.length,
                       concatenate=args.concatenate,
                       sparse=args.sparse,
                       dtype=eval(args.dtype),
                       device=args.device)
        # elif args.file_type == "distance":
        #     return cls(in_files=in_files,
        #                out_file=args.out_file,
        #                wrapper=Distance,
        #                A=args.A,
        #                eps=args.eps,
        #                bins=eval(args.bins),
        #                # max_dist=args.max_dist,
        #                # bin_width=args.bin_width,
        #                dim=args.dim,
        #                length=args.length,
        #                dtype=eval(args.dtype),
        #                device=args.device)
        # elif args.file_type == "Zuker":
        #     return cls(in_files=in_files,
        #                out_file=args.out_file,
        #                wrapper=Zuker,
        #                dataset=ZukerDataset,
        #                dim=args.dim,
        #                length=args.length,
        #                dtype=eval(args.dtype),
        #                device=args.device)
