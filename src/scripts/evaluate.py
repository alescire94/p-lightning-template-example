import argparse
from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from src.data import Vocab
from src.pl_modules import BasePLModule


def decode(preds: List[int], vocab: Vocab) -> List[int]:
    return [vocab.id_to_token(pred) for pred in preds]


def encode_tokens(sentence: List[str], vocab: Vocab, padding_size) -> torch.LongTensor:
    result: List[int] = [vocab.token_to_id(token) if token in vocab else vocab.unk_id for token in sentence]
    len_sentence = len(sentence)
    if len_sentence < padding_size:
        padding: List[int] = [vocab.pad_id] * (padding_size - len_sentence)
        result.extend(padding)
    return torch.LongTensor(result)


@hydra.main(config_path=hydra.utils.to_absolute_path("conf"), config_name="root")
def evaluate(conf: omegaconf.DictConfig) -> None:
    # reproducibility
    pl.seed_everything(conf.train.seed)

    sentence = conf.evaluate.sentence
    model_checkpoint = conf.evaluate.model_checkpoint_path

    vocabs = torch.load(hydra.utils.to_absolute_path("data/vocabs.pth"))
    model_path = hydra.utils.to_absolute_path(model_checkpoint)
    model: BasePLModule = BasePLModule.load_from_checkpoint(model_path)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    tokens: List[str] = sentence.split(" ")
    len_sentence = len(tokens)
    softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        tok_ids = encode_tokens(tokens, vocabs["words"], conf.data.padding_size)
        logits = model.forward(tok_ids.unsqueeze(0).to(device))
        pred = torch.argmax(softmax(logits), dim=-1).squeeze(0).tolist()[:len_sentence]
        decoded_sentence = decode(pred, vocabs["ner_labels"])
        print(decoded_sentence)


if __name__ == "__main__":
    evaluate()
