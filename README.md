# Kirigami
RNA secondary structure prediction via deep learning.

## Requirements
* python (>=3.9)
* C++17 compatible compiler
* Docker
* Singularity

## Install
```
docker pull python
chmod +x run.sh
./run.sh pip install .
```

## Usage
Commands are run within the `python3.9` Singularity container as follows:
```
./run.sh kirigami $COMMAND $OPTION+
```
The arguments to the `kirigami` module itself are as follows:
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
