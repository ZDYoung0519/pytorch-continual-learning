import numpy as np
import torch
from matplotlib import pyplot as plt


def tensor2numpy(x):
    if isinstance(x, np.ndarray):
        return x
    elif isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    else:
        raise ValueError(f'Type of x: {type(x)} cannot be converted to numpy!')


def topk_accuracy(y_true, y_pred, topk=1):
    return (y_pred.T == np.tile(y_true, (topk, 1))).sum() / len(y_true)


