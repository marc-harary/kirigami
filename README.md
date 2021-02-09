# Kirigami
RNA secondary structure prediction via deep learning.

## Requirements
* python (>=3.8)
* C++17 compatible compiler
* virtualenv

## Install
```
git clone https://github.com/marc-harary/kirigami.git
cd kirigami
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
pip install .
```

## Usage
The arguments to the `kirigami` module are passed as follows:
```
usage: kirigami [-h] [--quiet QUIET] {embed,train,evaluate} ...

positional arguments:
  {embed,train,evaluate}
    embed               embed .bpseq files
    train               train network
    evaluate            evaluate network on test files

optional arguments:
  -h, --help            show this help message and exit
  --quiet QUIET, -q QUIET
                        quiet
```

## Credits
Thermodynamic subroutines sourced from https://github.com/keio-bioinformatics/mxfold2.
