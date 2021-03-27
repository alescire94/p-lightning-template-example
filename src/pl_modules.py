from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.metrics.functional import f1 as f1_score

import torch
from torch import nn


class BasePLModule(pl.LightningModule):
    def __init__(self, conf, training_dataset, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.conf = conf
        self.num_labels = training_dataset.vocab_ner_labels_size
        lstm_param = self.conf.model.lstm
        self.word_embeddings = nn.Embedding(training_dataset.vocab_words_size, lstm_param.input_size, padding_idx=0)

        self.lstm = nn.LSTM(**self.conf.model.lstm)
        linear_input_size = (
            self.conf.model.lstm.input_size * 2
            if self.conf.model.lstm.bidirectional
            else self.conf.model.lstm.input_size
        )
        self.linear = nn.Linear(linear_input_size, training_dataset.vocab_ner_labels_size)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)
        self.softmax = nn.Softmax(dim=-1)
    def forward(self, sample) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        emb = self.word_embeddings(sample)
        out_lstm, _ = self.lstm(emb)
        return self.linear(out_lstm)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss, f1 = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        self.log("train_f1", f1)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss, f1 = self._shared_step(batch, batch_idx)
        self.log("val_loss", loss)
        self.log("val_f1", f1)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        loss, f1 = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss)
        self.log("test_f1", f1)
        return loss

    def _shared_step(self, batch: dict, batch_idx: int):
        words = batch["words"]
        label = batch["labels"]
        forward_output = self.forward(words)
        loss, f1 = self._evaluate(forward_output, label)
        return loss, f1

    def _evaluate(self, logits, labels):
        pred = self.softmax(logits)
        pred = torch.argmax(pred, dim=-1)
        mask = (labels != 0)
        pred_no_pad, labels_no_pad = pred[mask], labels[mask]
        f1 = f1_score(pred_no_pad, labels_no_pad, num_classes=self.num_labels)
        loss = self.loss(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return loss, f1

    def configure_optimizers(self):
        """
        FROM PYTORCH LIGHTNING DOCUMENTATION

        Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        Return:
            Any of these 6 options.

            - Single optimizer.
            - List or Tuple - List of optimizers.
            - Two lists - The first list has multiple optimizers, the second a list of LR schedulers (or lr_dict).
            - Dictionary, with an 'optimizer' key, and (optionally) a 'lr_scheduler'
              key whose value is a single LR scheduler or lr_dict.
            - Tuple of dictionaries as described, with an optional 'frequency' key.
            - None - Fit will run without any optimizer.
        """
        optimizer = torch.optim.Adam(self.parameters())
        # optimizer: torch.optim.Adam = hydra.utils.instantiate(self.conf.train.optimizer, params=self.parameters())
        return optimizer
