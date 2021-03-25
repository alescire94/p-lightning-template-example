import torch
from omegaconf import DictConfig


class AdamOptimizer(torch.optim.Adam):

    def __init__(self, conf: DictConfig):
        self.conf = conf
        super().__init__()
