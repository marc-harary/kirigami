import sys
import pathlib
import argparse
from kirigami.scripts import *

def main():
    parser = argparse.ArgumentParser(prog='kirigami')
    parser.add_argument('--quiet', '-q', type=bool, help='quiet', default=False)
    subparsers = parser.add_subparsers(dest='command', required=True)

    parser_embed = subparsers.add_parser('embed', help='embed .bpseq files')
    parser_embed.add_argument('--in-list', required=True, type=pathlib.Path, help='path to list file')
    parser_embed.add_argument('--out-directory', required=True, type=pathlib.Path, help='path to output directory')
    parser_embed.set_defaults(func=embed)

    parser_train = subparsers.add_parser('train', help='train network')
    parser_train.add_argument('--config', required=True, type=pathlib.Path, help='path to config file')
    parser_train.set_defaults(func=train)

    parser_evaluate = subparsers.add_parser('evaluate', help='train network files')
    parser_evaluate.add_argument('--config', required=True, type=pathlib.Path, help='path to config file')
    parser_evaluate.add_argument('--in-list', required=True, type=pathlib.Path, help='path to list file')
    parser_evaluate.set_defaults(func=evaluate)

    args = parser.parse_args()
    print(args)
    args.func(args)

if __name__ == '__main__':
    main()
