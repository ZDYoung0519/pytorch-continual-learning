import glob
import os
import numpy as np
import warnings
from tqdm import tqdm

import torch
from torch import nn
from torch.utils.data import DataLoader

from .base_learner import BaseLeaner


class EWC(BaseLeaner):
    def __init__(self, cfg, model, datasets, logger, lamb, importance):
        super(EWC, self).__init__(cfg, model, datasets, logger)
        self.lamb = lamb
        self.importance = importance

    def criterion(self, outputs, targets):
        loss_ce = nn.CrossEntropyLoss()(outputs, targets)
        if self.cur_t == 0:
            loss_reg = 0.
        else:
            loss_reg = 0.
        loss = loss_ce + self.lamb * loss_reg
        return loss







