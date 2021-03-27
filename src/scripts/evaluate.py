import torch
import hydra
from torch.utils.data import DataLoader

import src.pl_modules as module

import hydra
import omegaconf
import pytorch_lightning as pl
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from src.data.CoNLLDataset import CoNLLDataset
from src.pl_data_modules import BasePLDataModule
from src.pl_modules import BasePLModule


def decode(preds, vocab):
    print(vocab)
    return [vocab.id_to_token(pred)[0] for pred in preds]


@hydra.main(config_path=hydra.utils.to_absolute_path("conf"), config_name="root")
def evaluate(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    pl.seed_everything(conf.train.seed)

    # data module declaration
    vocabs = torch.load(hydra.utils.to_absolute_path("data/vocabs.pth"))
    print(type(vocabs['ner_labels']))
    vocab_words_size = len(vocabs["words"])
    vocab_pos_size = len(vocabs["vocab_pos"])
    vocab_ner_labels_size = len(vocabs["ner_labels"])
    vocab_sizes = {"words": vocab_words_size, "pos": vocab_pos_size,
                   "ner_labels": vocab_ner_labels_size}  # main module declaration
    model_path = hydra.utils.to_absolute_path(
        'experiments/default_name/2021-03-27/19-25-39/lightning_logs/version_0/checkpoints/epoch=0-step=234.ckpt')
    model = BasePLModule.load_from_checkpoint(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # model = pl_module.load_state_dict(torch.load(model_path,
    #                                              map_location=device))
    model.to(device)
    model.eval()
    dataset = CoNLLDataset(150, hydra.utils.to_absolute_path(conf.data.test_path))
    data_loader = DataLoader(dataset, num_workers=conf.data.num_workers, batch_size=conf.data.batch_size,
                      shuffle=False)

    with torch.no_grad():
        for x in data_loader:
            x = {k: v.to(device) if not isinstance(v, list) else v for k, v in x.items()}
            for i in range(conf.data.batch_size):
                decoded = decode(x['labels'][i], vocabs['ner_labels'])
                print(decoded)
                exit(0)


if __name__ == "__main__":
    evaluate()
