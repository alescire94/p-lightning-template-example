from typing import List

import hydra
import omegaconf
import pytorch_lightning as pl
import torch

from src.data import Vocab
from src.pl_modules import BasePLModule

"""
The evaluate script takes a checkpoint of your model and a sentence, to produce a NER labeling of that sentence.
"""


def decode(preds: List[int], vocab: Vocab) -> List[int]:
    return [vocab.id_to_token(pred) for pred in preds]


def encode_tokens(sentence: List[str], vocab: Vocab) -> torch.LongTensor:
    result: List[int] = [vocab.token_to_id(token) if token in vocab else vocab.unk_id for token in sentence]
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
    softmax = torch.nn.Softmax(dim=-1)
    with torch.no_grad():
        encoded_sentence = encode_tokens(tokens, vocabs["words"])
        logits = model.forward(encoded_sentence.unsqueeze(0).to(device))
        pred = torch.argmax(softmax(logits), dim=-1).squeeze(0).tolist()
        decoded_sentence = decode(pred, vocabs["ner_labels"])
        print(tokens)
        print(decoded_sentence)


if __name__ == "__main__":
    evaluate()
