from typing import *
from pathlib import Path
from dataclasses import dataclass
from copy import copy
import os
from collections import namedtuple

import torch

from kirigami.containers.fasta import *
from kirigami.containers.contact import *
from kirigami.containers.distance import *
from kirigami.containers.zuker import *


__all__ = ["Molecule"]


# @dataclass
class Molecule:
    _fasta: Fasta
    _contact: Optional[Contact]
    _distance: Optional[Distance]
    _zuker: Optional[Zuker]
    _name: Optional[str]

    out_tuple: NamedTuple = namedtuple("out_tuple", field_names=["ipt","contact","distance"]) #"inv_dist","bin_dist"])
    
    def __init__(self,
                 fasta: Fasta,
                 contact: Optional[Contact] = None,
                 distance: Optional[Distance] = None,
                 zuker: Optional[Zuker] = None,
                 name: Optional[str] = None) -> None:
        self._fasta = fasta
        self._contact = contact
        self._distance = distance
        self._zuker = zuker
        self._name = name

    def __len__(self) -> int:
        return len(self._fasta.to_str())

    def to_sparse(self) -> "Molecule":
        new_fasta = self.fasta.to_sparse()
        new_contact = None if self._contact is None else self._contact.to_sparse()
        new_distance = None if self._distance is None else self._distance.to_sparse()
        new_zuker = None if self._zuker is None else self._zuker.to_sparse()
        return type(self)(new_fasta, new_contact, new_distance, new_zuker, self._name)

    def to_dense(self) -> "Molecule":
        new_fasta = self.fasta.to_dense()
        new_contact = None if self._contact is None else self._contact.to_dense()
        new_distance = None if self._distance is None else self._distance.to_dense()
        new_zuker = None if self._zuker is None else self._zuker.to_dense()
        return type(self)(new_fasta, new_contact, new_distance, new_zuker, self._name)

    def float(self) -> "Molecule":
        new_fasta = self.fasta.float()
        new_contact = None if self._contact is None else self._contact.float()
        new_distance = None if self._distance is None else self._distance.float()
        new_zuker = None if self._zuker is None else self._zuker.float()
        return type(self)(new_fasta, new_contact, new_distance, new_zuker, self._name)

    def data(self,
             dtype: Optional[torch.dtype] = None,
             device: torch.device = torch.device("cpu")) -> Tuple[torch.Tensor,...]:
        out = []
        if not self.fasta.is_concat:
            new_fasta = self.fasta.concat()
        if self._zuker is not None:
            new_fasta = torch.stack((new_fasta.float(), self._zuker), dim=-3)
        out.append(new_fasta)
        if self._contact is not None:
            out.append(self._contact)
        if self._distance is not None:
            out.append(self._distance)
        for i in range(len(out)):
            if dtype is not None and out[i].dtype != dtype:
                out[i] = out[i].to(dtype)
            if out[i].device != device:
                out[i] = out[i].to(device)
        return tuple(out)
    
    @property
    def bases(self) -> str:
        return self._fasta.to_str()

    @property
    def ipt(self) -> torch.Tensor:
        if self._zuker is not None:
            return torch.cat((self._fasta.concat(), self._zuker), dim=-3)
        return self._fasta.concat()

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, val: str) -> None:
        self._name = val

    @property
    def fasta(self) -> Fasta:
        return self._fasta

    @fasta.setter
    def fasta(self, val: Fasta) -> None:
        self._fasta = val

    @property
    def contact(self) -> Contact:
        return self._contact

    @contact.setter
    def contact(self, val: Contact) -> None:
        self._contact = val

    @property
    def distance(self) -> Distance:
        return self._distance

    @distance.setter
    def distance(self, val: Distance) -> None:
        self._distance = val

    @property
    def zuker(self) -> Zuker:
        return self._zuker

    @zuker.setter
    def zuker(self, val: Zuker) -> None:
        self._zuker = val

    @staticmethod
    def _dotbracket2dict(dot_bracket: str) -> Dict[int,int]:
        paren = deque()
        square = deque()
        curly = deque()
        out = {}
        for i, char in enumerate(lines):
            if char == "(":
                paren.append(i)
            elif char == ")":
                j = paren.pop()
                out[i] = j
                out[j] = i
            elif char == "[":
                square.append(i)
            elif char == "]":
                j = square.pop()
                out[i] = j
                out[j] = i
            elif char == "{":
                curly.append(i)
            elif char == "}":
                j = curly.pop()
                out[i] = j
                out[j] = i
        return out
        
    @classmethod
    def from_bpseqs(cls, bpseq: str) -> "Molecule":
        lines = copy(bpseq).splitlines()
        lines = list(filter(lambda line: not line.startswith("#"), lines))
        # pairs = defaultdict(lambda: None)
        pairs = {}
        length = len(lines)
        fasta = ""
        for line in lines:
            i, base, j = line.split()
            i, j = int(i) - 1, int(j) - 1
            fasta += base.upper()
            if j == -1:
                continue
            # pairs[i], pairs[j] = j, i
            pairs[i] = j
        fasta_tensor = Fasta.from_str(fasta)
        # con = Contact.from_dict(pairs)#, length=len(pairs))  
        con = torch.zeros(len(fasta), len(fasta), dtype=torch.uint8)
        for i, j in pairs.items():
            if j != -1:
                # con[i,j] = con[j,i] = 1 
                con[i,j] = 1 
                con[j,i] = 1
        return cls(fasta_tensor, con)

    @classmethod
    def from_fastas(cls, fasta: str) -> "Molecule":
        fasta = copy(fasta).splitlines()
        chars = fasta[-1]
        fasta = Fasta.from_str(chars)
        return cls(fasta)

    @classmethod
    def from_sts(cls, st: str) -> "Molecule":
        start_idx = 0
        while lines[start_idx].startswith("#"):
            start_idx += 1
        fasta = lines[start_idx]
        dot_bracket = lines[start_idx+1]
        contact_map = cls.dotbracket2dict(dot_bracket)
        return cls(contact_map, fasta)

    @classmethod
    def from_zukers(cls, zuker: str) -> "Molecule":
        lines = copy(zuker).splitlines()
        lines = list(filter(lambda line: not line.startswith("#"), lines))
        contact_pairs = {}
        length = len(lines)
        fasta = ""
        zuker_pairs = {}
        for line in lines:
            words = line.split()
            if words[1].isalpha(): # in regular bpseq-portion of file
                ii, base, jj = words
                ii, jj = int(ii) - 1, int(jj) - 1
                fasta += base.upper()
                if jj == -1:
                    continue
                contact_pairs[ii], contact_pairs[jj] = jj, ii
            else: # in Zuker-portion of file
                prob = float(words[2]) 
                if prob == 0:
                    continue
                ii, jj = list(map(int, words[:2]))
                zuker_pairs[ii, jj] = prob
        fasta = Fasta.from_str(fasta)
        con = Contact.from_dict(contact_pairs)
        zuker = Zuker.from_dict(zuker_pairs)
        return cls(fasta, con, zuker)

    @classmethod
    def from_file(cls, file_path: Path, store_name: bool = True) -> "Molecule":
        with open(file_path, "r") as f:
            txt = f.read().strip()
        if file_path.endswith("fasta"):
            out = cls.from_fastas(txt)
        elif file_path.endswith("bpseq"):
            out = cls.from_bpseqs(txt)
        elif file_path.endswith("zuker"):
            out = cls.from_zukers(txt)
        elif file_path.endswith("st"):
            out = cls.from_sts(txt)
        else:
            raise ValueError("Invalid file type") 
        if store_name:
            base_name = os.path.basename(file_path)
            base_name, _ = os.path.splitext(base_name)
            out.name = base_name 
        return out
