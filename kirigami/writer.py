from typing import List
from pytorch_lightning.callbacks import BasePredictionWriter
from pytorch_lightning import LightningModule, Trainer
import torch
from tqdm import tqdm
from kirigami.utils import mat2db


class DbnWriter(BasePredictionWriter):
    def __init__(self):
        super().__init__(write_interval="epoch")

    def write_on_batch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        prediction: torch.Tensor,
        batch_indices: torch.Tensor,
        batch: torch.Tensor,
        batch_idx: int,
        dataloader_idx: int,
    ) -> None:
        opt_dir = trainer.datamodule.output_dir
        mol = trainer.datamodule.predict_mols[batch_idx]
        fasta = trainer.datamodule.predict_fastas[batch_idx]
        opt_file = opt_dir / mol.with_suffix(".dbn")
        dbn = mat2db(prediction)
        opt_file.write_text(f">{mol}\n{fasta}\n{dbn}\n")

    def write_on_epoch_end(
        self,
        trainer: Trainer,
        pl_module: LightningModule,
        predictions: List[torch.Tensor],
        batch_indices: torch.Tensor,
    ) -> None:
        opt_dir = trainer.datamodule.output_dir
        for batch_idx in tqdm(batch_indices[0]):
            batch_idx = batch_idx[0]
            mol = trainer.datamodule.predict_mols[batch_idx]
            fasta = trainer.datamodule.predict_fastas[batch_idx]
            prd = predictions[batch_idx]
            dbn = mat2db(prd)
            opt_file = (opt_dir / mol).with_suffix(".dbn")
            opt_file.write_text(f">{mol}\n{fasta}\n{dbn}\n")
