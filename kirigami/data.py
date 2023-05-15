from pathlib import Path
from typing import Optional, Tuple, Union
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from kirigami.utils import embed_fasta, embed_dbn


class DataModule(pl.LightningDataModule):
    """
    Kirigami API for reading, writing, and embedding data.

    Attributes
    ----------
    bprna_dir : `pathlib.Path`
        Path to directory containing bpRNA data.
    input_dir : `pathlib.Path`
        Path to directory for input FASTA files for prediction.
    output_dir : `pathlib.Path`
        Path to directory for output DBN files for prediction.
    train_path : `pathlib.Path`
        Path to bpRNA DBN files for training.
    val_path : `pathlib.Path`
        Path to bpRNA DBN files for validation.
    test_path : `pathlib.Path`
        Path to bpRNA DBN files for testing.
    train_dataset : list
        List of embedded bpRNA samples for training.
    val_dataset : list
        List of embedded bpRNA samples for validation.
    test_dataset : list
        List of embedded bpRNA samples for testing.
    predict_dataset : list
        List of embedded FASTA samples for prediction.
    predict_mols : list
        List of names of FASTA samples for prediction.
    predict_fastas : list
        List of original FASTA strings for prediction.

    Methods
    -------
    prepare_data()
        Embeds bpRNA files if not already embedded.
    setup(stage: str)
        Reads embedded data from disk for corresponding stage of Kirigami
        pipeline and stores in corresponding dataset attribute.
    train_dataloader()
        Returns `torch.data.Dataloader` object for training dataset.
    val_dataloader()
        Returns `torch.data.Dataloader` object for validation dataset.
    test_dataloader()
        Returns `torch.data.Dataloader` object for test dataset.
    predict_dataloader()
        Returns `torch.data.Dataloader` object for prediction dataset.
    _collate_fn(batch: torch.Tensor)
        Collation function for `Dataloader`s; performs outer-concatenation and
        one-hot embedding.
    """

    def __init__(
        self,
        bprna_dir: Optional[Path] = None,
        input_dir: Optional[Path] = None,
        output_dir: Optional[Path] = None,
    ):
        super().__init__()

        self.bprna_dir = bprna_dir or Path.cwd() / "data" / "bpRNA"
        self.input_dir = input_dir or Path.cwd() / "data" / "predict" / "input"
        self.output_dir = output_dir or Path.cwd() / "data" / "predict" / "output"

        for folder in [self.bprna_dir, self.input_dir, self.output_dir]:
            if not folder.exists():
                folder.mkdir()

        self.train_path = self.bprna_dir / "TR0.pt"
        self.val_path = self.bprna_dir / "VL0.pt"
        self.test_path = self.bprna_dir / "TS0.pt"

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.predict_dataset = None

        self.predict_mols = None
        self.predict_fastas = None

    def prepare_data(self):
        if not self.train_path.exists():
            train_list = embed_dbn(self.bprna_dir / "TR0.dbn")
            torch.save(train_list, self.train_path)

        if not self.val_path.exists():
            val_list = embed_dbn(self.bprna_dir / "VL0.dbn")
            torch.save(val_list, self.val_path)

        if not self.test_path.exists():
            test_list = embed_dbn(self.bprna_dir / "TS0.dbn")
            torch.save(test_list, self.test_path)

    def setup(self, stage: str):
        self.train_dataset = torch.load(self.train_path)
        self.val_dataset = torch.load(self.val_path)
        self.test_dataset = torch.load(self.test_path)
        if stage == "predict":
            self.predict_mols = []
            self.predict_fastas = []
            self.predict_dataset = []
            for file in tqdm(sorted(self.input_dir.iterdir())):
                mol, fasta, data = embed_fasta(file)
                self.predict_mols.extend(mol)
                self.predict_fastas.extend(fasta)
                self.predict_dataset.extend(data)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            collate_fn=self._collate_fn,
            shuffle=True,
            pin_memory=True,
            batch_size=1,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            collate_fn=self._collate_fn,
            shuffle=False,
            batch_size=1,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            collate_fn=self._collate_fn,
            shuffle=False,
            batch_size=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.predict_dataset,
            collate_fn=self._collate_fn,
            shuffle=False,
            batch_size=1,
        )

    def _collate_fn(
        self, batch: torch.Tensor
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Collation function for `Dataloader`s; performs outer-concatenation and
        one-hot embedding.

        Parameters
        ----------
        batch : torch.Tensor
            Tensor corresponding to input FASTA file and, if applicable,
            ground truth adjacency matrix.

        Returns
        -------
        fasta : torch.Tensor
            Embedded FASTA string.
        con : torch.Tensor
            Ground truth adjacency matrix if applicable.
        """
        (batch,) = batch
        fasta = batch[0]
        fasta = fasta[..., None]
        fasta = torch.cat(fasta.shape[-2] * [fasta], dim=-1)
        fasta_t = fasta.transpose(-1, -2)
        fasta = torch.cat([fasta, fasta_t], dim=-3)
        fasta = fasta[None, ...]
        fasta = fasta.float()
        if len(batch) == 1:
            return fasta
        con = batch[1]
        con = con[None, None, ...]
        con = con.float()
        return fasta, con
