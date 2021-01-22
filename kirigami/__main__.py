import pathlib
import argparse
from scripts.embed import embed
from scripts.train import train
from scripts.evaluate import evaluate

def main():
    parser = argparse.ArgumentParser(prog='kirigami')
    parser.add_argument('--quiet', '-q', type=bool, help='quiet', default=False)
    subparsers = parser.add_subparsers()

    parser_embed = subparsers.add_parser('embed', help='embed .bpseq files')
    parser_embed.add_argument('--in-list', type=pathlib.Path, help='path to list file') 
    parser_embed.add_argument('--out-directory', type=pathlib.Path, help='path to output directory')
    parser_embed.set_defaults(func=embed)

    parser_train = subparsers.add_parser('train', help='train network')
    parser_train.add_argument('--config', type=pathlib.Path, help='path to config file') 
    parser_train.set_defaults(func=train)

    parser_test = subparsers.add_parser('evaluate', help='train network files')
    parser_test.add_argument('--config', type=pathlib.Path, help='path to config file') 
    parser_test.add_argument('--in-list', type=pathlib.Path, help='path to list file')
    parser_train.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)

if __name__ == '__main__':
    main()
