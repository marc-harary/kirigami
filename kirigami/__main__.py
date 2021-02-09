from pathlib import Path
import argparse
from kirigami.scripts import *


def main():
    parser = argparse.ArgumentParser(prog='kirigami')
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_predict = subparsers.add_parser('predict', help='predict structure of `FASTA` files')
    parser_predict.add_argument('--quiet', '-q', type=bool, help='quiet', default=False)
    parser_predict.add_argument('--disable-cuda', type=bool, help='Disable CUDA', default=False)
    parser_predict.add_argument('--config', required=True, type=Path, help='path to config file')
    parser_predict.add_argument('--in-list', required=True, type=Path,
                                help='path to input list file of `.fasta`\'s')
    parser_predict.add_argument('--out-list', required=True, type=Path,
                                help='path to output list file of `.bpseq`\'s')
    parser_predict.add_argument('--out-directory', required=True, type=Path,
                                help='path to output directory of `.bpseqs`\'s')
    parser_predict.set_defaults(func=predict)

    parser_train = subparsers.add_parser('train', help='train network')
    parser_train.add_argument('--quiet', '-q', type=bool, help='quiet', default=False)
    parser_train.add_argument('--disable-cuda', type=bool, help='Disable CUDA', default=False)
    parser_train.add_argument('--config', required=True, type=Path, help='path to config file')
    parser_train.add_argument('--resume', required=False, type=bool, default=False,
                              help='resume training using config file')
    parser_train.set_defaults(func=train)

    parser_evaluate = subparsers.add_parser('evaluate', help='evaluate network on test files')
    parser_evaluate.add_argument('--quiet', '-q', type=bool, help='quiet', default=False)
    parser_evaluate.add_argument('--disable-cuda', type=bool, help='Disable CUDA', default=False)
    parser_evaluate.add_argument('--config', required=True, type=Path, help='path to config file')
    parser_evaluate.add_argument('--in-list', required=True, type=Path,
                                 help='path to input list file of `.bpseqs`\'s')
    parser_evaluate.add_argument('--out-directory', required=True, type=Path,
                                 help='path to output directory of `.fastas`\'s and `.bpseq`\'s')
    parser_evaluate.add_argument('--thres', type=float, default=.5,
                                 help='threshhold for binarizing output file')
    parser_evaluate.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == '__main__':
    main()
