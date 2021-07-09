import sys
from pathlib import Path
import argparse
from kirigami.core.embed import *
from kirigami.core.train import *
from kirigami.core.test import *
# from kirigami.predict import *


def main():
    parser = argparse.ArgumentParser(prog="kirigami", fromfile_prefix_chars="@")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # parser_predict = subparsers.add_parser("predict", help="predict structure of `FASTA` files")
    # parser_predict.add_argument("--config", "-c",
    #                             required=True,
    #                             type=Path,
    #                             help="path to config file")
    # parser_predict.add_argument("--in-list", "-i",
    #                             required=True,
    #                             type=Path,
    #                             help="path to input list file of `.fasta`\"s")
    # parser_predict.add_argument("--out-list",
    #                             required=True,
    #                             type=Path,
    #                             help="path to output list file of `.bpseq`\"s")
    # parser_predict.add_argument("--out-directory",
    #                             required=False,
    #                             default=".",
    #                             type=Path,
    #                             help="path to output directory of `.bpseqs`\"s")
    # parser_predict.add_argument("--log-file", "-l",
    #                             default=None,
    #                             type=Path,
    #                             help="path to log file")
    # parser_predict.add_argument("--quiet", "-q",
    #                             action="store_true",
    #                             help="quiet")
    # parser_predict.add_argument("--disable-cuda", "-d",
    #                             action="store_true",
    #                             help="Disable CUDA") 
    # parser_predict.set_defaults(func = lambda args: predict(config=path2munch(args.config),
    #                                                         in_list=args.in_list,
    #                                                         out_list=args.out_list,
    #                                                         out_dir=args.out_directory,
    #                                                         log_file=args.log_file,
    #                                                         quiet=args.quiet,
    #                                                         disable_cuda=args.disable_cuda)) 

    train_parser = subparsers.add_parser("train", help="train network")
    train_parser.add_argument("--training-checkpoint-file",
                              required=True,
                              type=Path,
                              help="path to training checkpoint file")
    train_parser.add_argument("--validation-checkpoint-file", 
                              default=None,
                              help="path to cross-validation checkpoint file",
                              type=Path)
    train_parser.add_argument("--log-file",
                              default=None,
                              type=Path,
                              help="path to log file")
    train_parser.add_argument("--resume",
                              default=False,
                              action="store_true", 
                              help="resume training")
    train_parser.add_argument("--quiet",
                              action="store_true",
                              help="quiet")
    train_parser.add_argument("--disable-batch-bar",
                              default=False,
                              action="store_true",
                              help="show tqdm progress bar for each batch")
    train_parser.add_argument("--disable-epoch-bar",
                              default=False,
                              action="store_true",
                              help="show tqdm progress bar for each epoch")
    train_parser.add_argument("--disable-cuda",
                              action="store_true",
                              help="disable CUDA")
    train_data = train_parser.add_argument_group()
    train_data.add_argument("--training-file",
                            required=True,
                            type=Path, 
                            help="path to dataset file")
    train_data.add_argument("--training-filetype",
                            choices=["bpseq-lst","pt-lst","st-lst","pt"],
                            type=str,
                            help="file type of training set")
    train_data.add_argument("--validation-file",
                            type=Path,
                            help="path to list of validation set")
    train_data.add_argument("--validation-filetype",
                            choices=["bpseq-lst","pt-lst","st-lst","pt"],
                            type=str,
                            help="file type of validation set")
    train_data.add_argument("--shuffle",
                            type=bool,
                            default=True,
                            help="shuffle validation and training sets")
    train_data.add_argument("--batch-size",
                            type=int,
                            default=1,
                            help="batch size for training and validation sets")
    train_data.add_argument("--batch-load",
                            default=False,
                            action="store_true",
                            help="pre-load and pre-embed all files prior to training")
    train_data.add_argument("--max-length",
                            default=512,
                            type=int,
                            help="maximum length to which all sequenes will be padded")
    train_data.add_argument("--num-workers",
                            default=0,
                            type=int,
                            help="number of workers for data loader")
    train_data.add_argument("--training-data-device",
                            required=True,
                            choices=["cpu","cuda"],
                            help="store training files on CPU but train on GPU")
    train_data.add_argument("--validation-data-device",
                            required=True,
                            choices=["cpu","cuda"],
                            help="store validation files on CPU and validate on GPU")
    train_data.add_argument("--disable-pre-concatenation",
                            default=False,
                            action="store_true",
                            help="skip pre-concatenation before input to network")
    train_parser.add_argument("--mixed-precision",
                              action="store_true",
                              default=False,
                              help="training precision")
    train_parser.add_argument("--epochs",
                              type=int,
                              required=True,
                              help="total epochs for training")
    train_parser.add_argument("--iters-to-accumulate",
                              type=int,
                              default=1,
                              help="number of batches between each optimizer step")
    train_parser.add_argument("--criterion",
                              type=str,
                              required=True,
                              help="criterion (loss function) for training")
    train_parser.add_argument("--optimizer",
                              type=str,
                              required=True,
                              help="optimizer algorithm for training")
    train_model = train_parser.add_argument_group()
    train_model.add_argument("--add-layer",
                             metavar="LAYER",
                             required=True,
                             action="append",
                             dest="layers",
                             type=str,
                             help="add layers to model")
    train_model.add_argument("--model-device",
                             required=True,
                             choices=["cpu","cuda"],
                             help="store model on CPU or GPU") 
    train_model.add_argument("--checkpoint-segments",
                             default=0,
                             type=int,
                             help="chunks in model for gradient checkpointing") 
    train_post_processing = train_parser.add_argument_group()
    train_post_processing.add_argument("--binarize",
                                       default=False,
                                       action="store_true",
                                       help="binarize and threshold matrix")
    train_post_processing.add_argument("--thres-prob",
                                       default=0.0,
                                       type=float,
                                       help="threshold probability for binarization")
    train_post_processing.add_argument("--thres-by-ground-pairs",
                                       default=False,
                                       action="store_true",
                                       help="select number of pairs as in ground truth") 
    train_post_processing.add_argument("--disable-canonicalize",
                                       default=True,
                                       action="store_false",
                                       help="filter for only Wobble and W-C pairs") 
    train_post_processing.add_argument("--disable-symmetrize",
                                       default=True,
                                       action="store_false",
                                       help="make output label matrix symmetrical")
    train_parser.set_defaults(func = lambda namespace: Train.from_namespace(namespace).run(namespace.resume))

    test_parser = subparsers.add_parser("test", help="test network on files")
    test_parser.add_argument("--out-file",
                             required=True,
                             type=Path,
                             help="path to output `.csv` file")
    test_parser.add_argument("--checkpoint-file",
                             required=True,
                             type=Path,
                             help="path to checkpoint file")
    test_parser.add_argument("--disable-sequence-bar",
                             default=False,
                             action="store_true",
                             help="show tqdm progress bar for each sequence")
    test_parser.add_argument("--criterion",
                             type=str,
                             required=True,
                             help="criterion (loss function) for training")
    test_model = test_parser.add_argument_group()
    test_parser.add_argument("--model-device",
                             required=True,
                             type=str,
                             help="device on which to test model")
    test_model.add_argument("--add-layer",
                            metavar="LAYER",
                            required=True,
                            action="append",
                            dest="layers",
                            type=str,
                            help="add layers to model")
    test_data = test_parser.add_argument_group()
    test_data.add_argument("--test-file",
                           required=True,
                           type=Path, 
                           help="path to test set file")
    test_data.add_argument("--test-filetype",
                           choices=["bpseq-list","pt-list", "st-list", "pt"],
                           type=str,
                           help="file type of test set")
    test_post_processing = test_parser.add_argument_group()
    test_post_processing.add_argument("--thres-prob",
                                      default=0.0,
                                      type=float,
                                      help="threshold probability for binarization")
    test_post_processing.add_argument("--disable-canonicalize",
                                      default=True,
                                      action="store_false",
                                      help="filter for only Wobble and W-C pairs") 
    test_post_processing.add_argument("--disable-symmetrize",
                                      default=True,
                                      action="store_false",
                                      help="make output label matrix symmetrical")
    test_parser.set_defaults(func = lambda namespace: Test.from_namespace(namespace).run())

    embed_parser = subparsers.add_parser("embed", help="embed various files")
    embed_parser.add_argument("--file-type",
                              choices=["contact", "distance"],
                              required=True,
                              type=str,
                              help="type of file to embed")
    embed_input = embed_parser.add_mutually_exclusive_group()
    embed_input.add_argument("--in-directory",
                             type=Path,
                             default=None,
                             help="directory of files to embed")
    embed_input.add_argument("--in-list",
                             type=Path,
                             default=None,
                             help="path to list file")
    embed_parser.add_argument("--length",
                              default=512,
                              type=int,
                              help="length to pad tensor")
    embed_parser.add_argument("--out-file",
                              default="out.pt",
                              type=Path,
                              help="path to output file")
    embed_parser.add_argument("--dim",
                              type=int,
                              default=3, 
                              help="dimensions of singleton tensor")
    embed_parser.add_argument("--A",
                              type=float,
                              default=1.0,
                              help="numerator for inverted tensor")
    embed_parser.add_argument("--eps",
                              type=float,
                              default=1e-4,
                              help="denominator offset")
    embed_parser.add_argument("--bins",
                              type=str,
                              default="torch.arange(4, 22.5, .5)",
                              help="bins for distance one-hot encoding")
    # embed_parser.add_argument("--max-dist",
    #                           type=float,
    #                           default=22.,
    #                           help="maximum distance for binning")
    # embed_parser.add_argument("--bin-width",
    #                           type=float,
    #                           default=.5,
    #                           help="bin width")
    embed_parser.add_argument("--device",
                              choices=["cpu", "cuda"],
                              type=str,
                              help="device to store data")
    embed_parser.add_argument("--sparse",
                              action="store_true",
                              help="embed tensors as sparse")
    embed_parser.add_argument("--dtype",
                              type=str,                              
                              default="torch.uint8",
                              help="datatype for tensor")
    embed_parser.set_defaults(func = lambda namespace: Embed.from_namespace(namespace).run())
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
