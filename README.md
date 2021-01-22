# Kirigami
RNA secondary structure prediction via deep learning.
## Requirements
* python (>=3.8)
* pytorch (>=1.7)
* C++17 compatible compiler
## Credits
Thermodynamic subroutines sourced from https://github.com/keio-bioinformatics/mxfold2.
## Usage
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
