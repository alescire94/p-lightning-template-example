from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn


class BasePLModule(pl.LightningModule):
    def __init__(self, conf, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.save_hyperparameters(conf)
        self.conf = conf
        self.word_embeddings = nn.Embedding(50_000, 300, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=300, hidden_size=300, bidirectional=False, batch_first=True, num_layers=300, dropout=0.2
        )
        self.linear = nn.Linear(300, 11)
        self.loss = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, feature) -> dict:
        """
        Method for the forward pass.
        'training_step', 'validation_step' and 'test_step' should call
        this method in order to compute the output predictions and the loss.

        Returns:
            output_dict: forward output containing the predictions (output logits ecc...) and the loss if any.

        """
        emb = self.word_embeddings(feature)
        out_lstm, _ = self.lstm(emb)
        return self.linear(out_lstm)

    def training_step(self, batch: dict, batch_idx: int) -> torch.Tensor:
        loss = self._shared_step(batch, batch_idx)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch: dict, batch_idx: int) -> None:
        loss = self._shared_step(batch, batch_idx)
        self.log("dev_loss", loss)
        return loss

    def test_step(self, batch: dict, batch_idx: int) -> Any:
        loss = self._shared_step(batch, batch_idx)
        self.log("test_loss", loss)
        return loss

    def _shared_step(self, batch: dict, batch_idx: int):
        words = batch["words"]
        label = batch["labels"]
        forward_output = self.forward(words)
        loss = self.loss(forward_output.view(-1, forward_output.shape[-1]), label.view(-1))
        return loss

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
