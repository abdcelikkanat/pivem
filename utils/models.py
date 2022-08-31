import torch
from utils.common import EPS


def remainder(x: torch.Tensor, y: float):

    remainders = torch.remainder(x, y)
    remainders[torch.abs(remainders - y) < EPS] = 0

    return remainders


def div(x: torch.Tensor, y: float, decimals=6):

    return torch.round(torch.div(torch.round(x, decimals=decimals), y, ), decimals=decimals).type(torch.int)


def vectorize(x: torch.Tensor):

    return x.flatten(-2)


def unvectorize(x: torch.Tensor, size):

    return x.reshape((size[0], size[1], size[2]))


def standardize(x: torch.Tensor):

    if x.dim() == 2:
        return x - torch.mean(x, dim=0, keepdim=True)

    elif x.dim() == 3:

        return x - torch.mean(x, dim=1, keepdim=True)

    else:

        raise ValueError("Input of the tensor must be 2 or 3!")


