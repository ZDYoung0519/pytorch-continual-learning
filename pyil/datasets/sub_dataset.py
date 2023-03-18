import numpy as np
import torch
from torch.utils.data import Dataset


class IDataset(Dataset):
    def __init__(self, dataset, tgt_classes, offset=0):

        self.tgt_classes = tgt_classes
        self.offset = offset

        self.indexes = np.where([y in tgt_classes for y in dataset.targets])[0]

        self.data = []
        self.targets = []
        for i in range(len(self.indexes)):
            image, target = dataset[self.indexes[i]]
            target = self.tgt_classes.index(target) + self.offset
            self.data.append(image)
            self.targets.append(target)

    def __getitem__(self, i):
        return self.data[i], self.targets[i]

    def __len__(self):
        return len(self.data)
