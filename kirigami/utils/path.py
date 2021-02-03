import os
import json
from pathlib import Path
from typing import List, Tuple
from munch import Munch


__all__ = ['path2munch', 'rename']


def path2munch(path: Path) -> munch.Munch:
    '''Reads .json file saved at PATH and returns `Munch` object'''
    with open(path, 'r') as f:
        txt = f.read()
    conf_json = json.loads(txt)
    conf =  munch.munchify(conf_json)
    return conf


def new_ext(in_file: Path, out_file: Path, out_dir: Path, ext: str) -> List[Path]:
    '''Reads `IN_FILE` and outputs list of new paths with new `EXT` at end'''
    out_files = []
    with open(in_file, 'r') as f:
        in_files = f.read().splitlines()
        for file in in_files:
            file = os.path.basename(file)
            file, _ = os.path.splitext(file)
            file += ext
            file = os.path.join(out_dir, file)
            out_files.append(file)
    with open(out_file, 'w') as f:
        for file in out_files:
            f.write(file+'\n')
    return out_files
