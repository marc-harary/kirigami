from typing import Union, List, Callable
from pathlib import Path
import argparse
import torch
from torch import nn


# class Train:
#     def __init__(self,
#                  layer_str: List[str],
#                  optimizer: torch.optim,
#                  criterion: Callable,
#                  training_set: Path,
#                  validation_set: Union[Path,None],
#                  log_file: Union[Path,None],
#                  checkpoint_path: Union[Path,None],
#                  batch_size: int = 1,
#                  shuffle: bool = True,
#                  disable_cuda: bool = False,
#                  quiet: bool = False):
# 
#         self.optimizer = optimizer
#         self.criterion = criterion
# 
#         self.training_set = training_set
#         self.validation_set = validation_set
# 
#         self.log_file = log_file
#         self.checkpoint_path = checkpoint_path
#         self.batch_size = batch_size
#         self.quiet = quiet
# 
#         if torch.cuda.is_available() and not disable_cuda:
#             self.device = torch.device("cuda")
#         else:
#             self.device = torch.device("cpu")
#         self.model.to(device)
#         
# 
#     @classmethod
#     def from_namespace(cls, args: argparse.Namespace):
#         model = nn.Sequential(*[eval(layer) for layer in args.layers])
#         optimizer = eval(optimizer if optimizer.startswith("torch.nn") else "torch.nn" + optimizer)
#         criterion = eval(criterion if criterion.startswith("torch.nn") else "torch.nn" + criterion)
#         training_set = 
#     
# 
#     def run(self, resume: bool = False):
#         if self.log_file:
#             logging.basicconfig(filename=log_file, level=logging.info)
#             logging.info("starting at " + str(datetime.datetime.now()))
#             
#         model.train()
#         start_epoch = 0
#         best_val_loss = float("inf")
# 
#         dataset = embeddeddataset if config.data.pre_embedded else bpseqdataset
#         train_set = dataset(config.data.training_list,
#                             device=device,
#                             quiet=quiet,
#                             batch_load=config.data.batch_load)
#         train_loader = dataloader(train_set,
#                                   batch_size=config.data.batch_size,
#                                   shuffle=config.data.shuffle,
#                                   num_workers=4)
# 
#         if config.data.validation_list:
#             val_set = dataset(config.data.validation_list,
#                               device=torch.device("cpu"),
#                               quiet=quiet,
#                               batch_load=config.data.batch_load)
#             val_loader = dataloader(val_set,
#                                     batch_size=config.data.batch_size,
#                                     shuffle=config.data.shuffle)
# 
#         if resume:
#             assert os.path.exists(config.training.checkpoint), "cannot find checkpoint file"
#             checkpoint = torch.load(config.training.checkpoint)
#             model.load_state_dict(checkpoint["model_state_dict"])
#             optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
#             start_epoch = checkpoint["epoch"]
#             loss = checkpoint["loss"]
#             if log_file:
#                 logging.info(f"resuming at epoch {epoch} with loss {loss}")
# 

def main():
    parser = argparse.ArgumentParser(fromfile_prefix_chars="@")
    # parser.add_argument("--model", "-m", dest="layers", action="append", type=lambda x: eval(x), required=True)
    parser.add_argument("--foo", type=lambda x: torch.load(x)) #, type=eval, required=True)
    args = parser.parse_args()

    import pdb; pdb.set_trace()
    
if __name__ == "__main__":
    main()
