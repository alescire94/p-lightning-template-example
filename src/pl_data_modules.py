from typing import Any, Union, List, Optional

import hydra
from omegaconf import DictConfig

import torch
from torch.utils.data import DataLoader, Dataset, IterableDataset
import pytorch_lightning as pl
import os

from src.data.CoNLLDataset import CoNLLDataset


class BasePLDataModule(pl.LightningDataModule):
    """
    FROM LIGHTNING DOCUMENTATION

    A DataModule standardizes the training, val, test splits, data preparation and transforms.
    The main advantage is consistent data splits, data preparation and transforms across models.

    Example::

        class MyDataModule(LightningDataModule):
            def __init__(self):
                super().__init__()
            def prepare_data(self):
                # download, split, etc...
                # only called on 1 GPU/TPU in distributed
            def setup(self):
                # make assignments here (val/train/test split)
                # called on every process in DDP
            def train_dataloader(self):
                train_split = Dataset(...)
                return DataLoader(train_split)
            def val_dataloader(self):
                val_split = Dataset(...)
                return DataLoader(val_split)
            def test_dataloader(self):
                test_split = Dataset(...)
                return DataLoader(test_split)

    A DataModule implements 5 key methods:

    * **prepare_data** (things to do on 1 GPU/TPU not on every GPU/TPU in distributed mode).
    * **setup**  (things to do on every accelerator in distributed mode).
    * **train_dataloader** the training dataloader.
    * **val_dataloader** the val dataloader(s).
    * **test_dataloader** the test dataloader(s).


    This allows you to share a full dataset without explaining how to download,
    split transform and process the data

    """

    def __init__(self, conf: DictConfig):
        super().__init__()
        self.conf = conf
        self.splits = ["train", "validation", "test"]

    # download dataset and stores it in train, val and test split in csv format.
    def prepare_data(self, *args, **kwargs):
        pass
        # dataset = load_dataset(self.conf.data.dataset_name)
        # for split in self.splits:
        #     split_path = hydra.utils.to_absolute_path(self.conf.data[f"{split}_path"])
        #     dataset[split].to_csv(split_path)

    def setup(self, stage: Optional[str] = None):
        # TODO os.system("wget data/ https://data.deepai.org/conll2003.zip && unzip data/conll2003.zip")
        # raise NotImplementedError
        pass

    def train_dataloader(self, *args, **kwargs) -> DataLoader:
        train_path = hydra.utils.to_absolute_path(self.conf.data.train_path)
        dataset = CoNLLDataset(self.conf.data.padding_size, train_path)
        return DataLoader(dataset, num_workers=self.conf.data.num_workers, batch_size=self.conf.data.batch_size)

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        val_path = hydra.utils.to_absolute_path(self.conf.data.validation_path)
        dataset = CoNLLDataset(self.conf.data.padding_size, val_path)
        return DataLoader(dataset, num_workers=self.conf.data.num_workers, batch_size=self.conf.data.batch_size)

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:
        test_path = hydra.utils.to_absolute_path(self.conf.data.test_path)
        dataset = CoNLLDataset(self.conf.data.padding_size, test_path)
        return DataLoader(dataset, num_workers=self.conf.data.num_workers, batch_size=self.conf.data.batch_size)

    def transfer_batch_to_device(self, batch: Any, device: torch.device) -> Any:
        pass
