from typing import Any, List, Optional, Union

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

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
        train_path = hydra.utils.to_absolute_path(self.conf.data.train_path)
        self.train_dataset = CoNLLDataset(self.conf.data.padding_size, train_path)
        val_path = hydra.utils.to_absolute_path(self.conf.data.validation_path)
        self.val_dataset = CoNLLDataset(self.conf.data.padding_size, val_path)
        test_path = hydra.utils.to_absolute_path(self.conf.data.test_path)
        self.test_dataset = CoNLLDataset(self.conf.data.padding_size, test_path)

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

        return DataLoader(
            self.train_dataset, num_workers=1, batch_size=self.conf.data.batch_size, shuffle=True
        )

    def val_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:

        return DataLoader(
            self.val_dataset, num_workers=1, batch_size=self.conf.data.batch_size, shuffle=False
        )

    def test_dataloader(self, *args, **kwargs) -> Union[DataLoader, List[DataLoader]]:

        return DataLoader(
            self.test_dataset, num_workers=1, batch_size=self.conf.data.batch_size, shuffle=False
        )
    '''
    def transfer_batch_to_device(self, batch, device):
        if isinstance(batch, CustomBatch):
            # move all tensors in your custom data structure to the device
            batch.samples = batch.samples.to(device)
            batch.targets = batch.targets.to(device)
        else:
            batch = super().transfer_batch_to_device(data, device)
        return batch
    '''