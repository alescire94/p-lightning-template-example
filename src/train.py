import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.pl_data_modules import BasePLDataModule
from src.pl_modules import BasePLModule


def train(conf: omegaconf.DictConfig) -> None:

    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    pl_data_module = BasePLDataModule(conf)

    dataset_train =  pl_data_module.train_dataset

    # main module declaration
    pl_module = BasePLModule(conf, dataset_train)

    # callbacks declaration
    callbacks_store = []

    if conf.train.early_stopping_callback is not None:
        early_stopping_callback: EarlyStopping = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(early_stopping_callback)

    if conf.train.model_checkpoint_callback is not None:
        model_checkpoint_callback: ModelCheckpoint = hydra.utils.instantiate(conf.train.early_stopping_callback)
        callbacks_store.append(model_checkpoint_callback)

    gpus = conf.train.pl_trainer.gpus if torch.cuda.is_available() else 0

    # trainer
    trainer: Trainer = hydra.utils.instantiate(
        conf.train.pl_trainer, callbacks=callbacks_store, gpus=gpus, fast_dev_run=False
    )

    # module fit
    trainer.fit(pl_module, datamodule=pl_data_module)

    # module test
    trainer.test(pl_module, datamodule=pl_data_module)


@hydra.main(config_path="../conf", config_name="root")
def main(conf: omegaconf.DictConfig):
    train(conf)


if __name__ == "__main__":
    main()
