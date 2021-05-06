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
                      default=False,
                      action="store_true",
                      help="pre-load and pre-embed all files prior to training")
    data.add_argument("--max-length",
                      default=512,
                      type=int,
                      help="maximum length to which all sequenes will be padded")
    data.add_argument("--num-workers",
                      default=0,
                      type=int,
                      help="number of workers for data loader")
    data.add_argument("--training-data-device",
                      required=True,
                      choices=["cpu","gpu"],
                      help="store training files on CPU but train on GPU")
    data.add_argument("--validation-data-device",
                      required=True,
                      choices=["cpu","gpu"],
                      help="store validation files on CPU and validate on GPU")

    parser_train.add_argument("--mixed-precision",
                              action="store_true",
                              default=False,
                              help="training precision")
    parser_train.add_argument("--epochs",
                              type=int,
                              required=True,
                              help="total epochs for training")
    parser_train.add_argument("--iters-to-accumulate",
                              type=int,
                              default=1,
                              help="number of batches between each optimizer step")
    parser_train.add_argument("--criterion",
                              type=str,
                              required=True,
                              help="criterion (loss function) for training")
    parser_train.add_argument("--optimizer",
                              type=str,
                              required=True,
                              help="optimizer algorithm for training")

    model = parser_train.add_argument_group()
    model.add_argument("--add-layer",
                       metavar="LAYER",
                       required=True,
                       action="append",
                       dest="layers",
                       type=str,
                       help="add layers to model")
    model.add_argument("--model-device",
                       required=True,
                       choices=["cpu","gpu"],
                       help="store model on CPU or GPU") 
    model.add_argument("--checkpoint-gradients",
                       default=False,
                       action="store_true",
                       help="enabled gradient checkpointing") 
    model.add_argument("--segments",
                       default=1,
                       type=int,
                       help="chunks in model for gradient checkpointing") 

    post_processing = parser_train.add_argument_group()
    post_processing.add_argument("--binarize",
                                 default=False,
                                 action="store_true",
                                 help="binarize and threshold matrix")
    post_processing.add_argument("--thres",
                                 default=0.5,
                                 type=float,
                                 help="threshold probability for binarization")
    post_processing.add_argument("--canonicalize",
                                 default=False,
                                 action="store_true",
                                 help="filter for only Wobble and W-C pairs") 
    post_processing.add_argument("--symmetrize",
                                 default=False,
                                 action="store_true",
                                 help="make output label matrix symmetrical")
    
                                
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
                              default=False,
                              help="show tqdm progress bar")
    parser_train.add_argument("--disable-cuda",
                              action="store_true",
                              help="disable CUDA")
    parser_train.set_defaults(func = lambda namespace: Train.from_namespace(namespace).run(namespace.resume))


    parser_test = subparsers.add_parser("evaluate", help="evaluate network on test files")
    test_model = parser_test.add_argument_group()
    test_model.add_argument("--add-layer",
                            metavar="LAYER",
                            required=True,
                            action="append",
                            dest="layers",
                            type=str,
                            help="add layers to model")
    test_data = parser_test.add_argument_group()
    test_data.add_argument("--test-file",
                           required=True,
                           type=Path, 
                           help="path to test set file")
    test_data.add_argument("--test-filetype",
                           choices=["bpseq-list","pt-list", "st-list"],
                           type=str,
                           help="file type of test set")
    parser_test.add_argument("--out-file",
                             required=True,
                             type=Path,
                             help="path to output `.csv` file")
    parser_test.add_argument("--thres",
                             nargs="?",
                             default=.5,
                             type=float,
                             help="threshhold for binarizing output file")
    parser_test.add_argument("--show_bar",
                             action="store_true",
                             help="show tqdm progress bar")
    parser_test.set_defaults(func = lambda namespace: Test.from_namespace(namespace).run())


    parser_embed = subparsers.add_parser("embed", help="embed various files")
    parser_embed.add_argument("--file-type",
                              choices=["bpseq", "st"],
                              required=True,
                              type=str,
                              help="type of file to embed")
    embed_input = parser_embed.add_mutually_exclusive_group()
    embed_input.add_argument("--in-directory",
                             type=Path,
                             help="directory of files to embed")
    embed_input.add_argument("--in-list",
                             type=Path,
                             help="path to list file")
    parser_embed.add_argument("--out-file",
                              default="out.pt",
                              type=Path,
                              help="path to output file")
    parser_embed.add_argument("--tensor-dim",
                              type=int,
                              default=3, 
                              help="dimensions of singleton tensor")
    parser_embed.add_argument("--concatenate",
                              action="store_true",
                              help="concatenate singletons")
    parser_embed.add_argument("--pad-length",
                              type=int,
                              default=512,
                              help="size to which to pad all sequences")
    parser_embed.add_argument("--device",
                              choices=["cpu", "cuda"],
                              type=str,
                              help="device to store data")
    parser_embed.add_argument("--sparse",
                              action="store_true",
                              help="embed tensors as sparse")
    parser_embed.set_defaults(func = lambda namespace: Embed.from_namespace(namespace).run())
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
