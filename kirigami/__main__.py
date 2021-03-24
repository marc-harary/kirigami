import sys
from pathlib import Path
import argparse
from kirigami.scripts import *


def main():
    parser = argparse.ArgumentParser(prog="kirigami", fromfile_prefix_chars="@")
    subparsers = parser.add_subparsers(dest="command", required=True)


    parser_predict = subparsers.add_parser("predict", help="predict structure of `FASTA` files")
    parser_predict.add_argument("--config", "-c",
                                required=True,
                                type=Path,
                                help="path to config file")
    parser_predict.add_argument("--in-list", "-i",
                                required=True,
                                type=Path,
                                help="path to input list file of `.fasta`\"s")
    parser_predict.add_argument("--out-list",
                                required=True,
                                type=Path,
                                help="path to output list file of `.bpseq`\"s")
    parser_predict.add_argument("--out-directory",
                                required=False,
                                default=".",
                                type=Path,
                                help="path to output directory of `.bpseqs`\"s")
    parser_predict.add_argument("--log-file", "-l",
                                default=None,
                                type=Path,
                                help="path to log file")
    parser_predict.add_argument("--quiet", "-q",
                                action="store_true",
                                help="quiet")
    parser_predict.add_argument("--disable-cuda", "-d",
                                action="store_true",
                                help="Disable CUDA") 
#     parser_predict.set_defaults(func = lambda args: predict(config=path2munch(args.config),
#                                                             in_list=args.in_list,
#                                                             out_list=args.out_list,
#                                                             out_dir=args.out_directory,
#                                                             log_file=args.log_file,
#                                                             quiet=args.quiet,
#                                                             disable_cuda=args.disable_cuda)) 
# 

    parser_train = subparsers.add_parser("train", help="train network")

    data = parser_train.add_argument_group()
    data.add_argument("--training-file",
                      required=True,
                      type=Path, 
                      help="path to dataset file")
    data.add_argument("--training-filetype",
                      choices=["bpseq-lst","pt-lst","st-lst","pt"],
                      type=str,
                      help="file type of training set")
    data.add_argument("--validation-file",
                      type=Path,
                      help="path to list of validation set")
    data.add_argument("--validation-filetype",
                      choices=["bpseq-lst","pt-lst","st-lst","pt"],
                      type=str,
                      help="file type of validation set")
    data.add_argument("--shuffle",
                      type=bool,
                      default=True,
                      help="shuffle validation and training sets")
    data.add_argument("--batch-size",
                      type=int,
                      default=1,
                      help="batch size for training and validation sets")
    data.add_argument("--batch-load",
                      action="store_true",
                      help="pre-load and pre-embed all files prior to training")
    data.add_argument("--copy-to-gpu",
                      action="store_true",
                      help="store files on CPU but train on GPU")

    parser_train.add_argument("--epochs",
                              type=int,
                              required=True,
                              help="total epochs for training")
    parser_train.add_argument("--criterion",
                              type=str,
                              required=True,
                              help="criterion (loss function) for training")
    parser_train.add_argument("--optimizer",
                              type=str,
                              required=True,
                              help="optimizer algorithm for training")
    parser_train.add_argument("--add-layer",
                              metavar="LAYER",
                              required=True,
                              action="append",
                              dest="layers",
                              type=str,
                              help="add layers to model")

    parser_train.add_argument("--training-checkpoint-file",
                              required=True,
                              type=Path,
                              help="path to training checkpoint file")
    parser_train.add_argument("--validation-checkpoint-file", 
                              default=None,
                              help="path to cross-validation checkpoint file",
                              type=Path)
    parser_train.add_argument("--log-file",
                              default=None,
                              type=Path,
                              help="path to log file")
    parser_train.add_argument("--resume",
                              action="store_true", 
                              help="resume training")
    parser_train.add_argument("--quiet",
                              action="store_true",
                              help="quiet")
    parser_train.add_argument("--show-bar",
                              action="store_true",
                              help="show tqdm progress bar")
    parser_train.add_argument("--disable-cuda",
                              action="store_true",
                              help="disable CUDA")
    parser_train.set_defaults(func = lambda namespace: Train.from_namespace(namespace).run(namespace.resume))


    parser_evaluate = subparsers.add_parser("evaluate", help="evaluate network on test files")
    model = parser_evaluate.add_argument_group()
    model.add_argument("--add-layer",
                       metavar="LAYER",
                       required=True,
                       action="append",
                       dest="layers",
                       type=str,
                       help="add layers to model")
    data = parser_evaluate.add_argument_group()
    data.add_argument("--test-file",
                      required=True,
                      type=Path, 
                      help="path to dataset file")
    data.add_argument("--test-filetype",
                      choices=["bpseq-list","pt-list", "st-list","pt"],
                      type=str,
                      help="file type of test set")
    parser_evaluate.add_argument("--out-directory",
                                 nargs="?",
                                 required=True,
                                 type=argparse.FileType("w"),
                                 default=sys.stdout,
                                 help="path to input list file of `.bpseqs`\"s")
    parser_evaluate.add_argument("--threshold",
                                 nargs="?",
                                 default=.5,
                                 type=float,
                                 help="threshhold for binarizing output file")
    parser_evaluate.add_argument("--log-file",
                                 default=sys.stdout,
                                 type=Path,
                                 help="path to log file")
    parser_evaluate.add_argument("--quiet",
                                 action="store_true",
                                 help="quiet")
    parser_evaluate.add_argument("--disable-cuda",
                                 action="store_true",
                                 help="disable CUDA")
    # parser_evaluate.set_defaults(func = lambda args: evaluate(config=munchify(args.config),
    #                                                           in_list=args.in_list,
    #                                                           out_dir=args.out_directory,
    #                                                           thres=args.thres,
    #                                                           log_file=args.log_file,
    #                                                           quiet=args.quiet,
    #                                                           disable_gpu=args.disable_gpu))


    parser_embed = subparsers.add_parser("embed", help="embed various files")
    parser_embed.add_argument("--file-type", "-f",
                              choices=["bpseq", "st"],
                              required=True,
                              type=str,
                              help="Type of file to embed")
    parser_embed.add_argument("--in-list", "-i",
                              required=True,
                              type=Path,
                              help="path to input list file")
    parser_embed.add_argument("--out-directory", "-o",
                              required=True,
                              default=".",
                              type=Path,
                              help="path to output directory of embedded files")
    parser_embed.add_argument("--log-file", "-l",
                              default=None,
                              help="path to log file") 
    parser_embed.add_argument("--quiet", "-q",
                              action="store_true",
                              help="quiet") 
#     parser_embed.set_defaults(func = lambda args: evaluate(file_type=args.file_type,
#                                                            in_list=args.in_list,
#                                                            out_dir=args.out_dir,
#                                                            log_file=args.log_file,
#                                                            quiet=args.quiet))
# 
# 
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
