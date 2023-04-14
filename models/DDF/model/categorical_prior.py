import torch
import torch.nn as nn
from torch.distributions import OneHotCategorical, Categorical

from model.network import NN


class CategoricalPrior(nn.Module):
    """
    Factorized Categorical prior with tunable logits.
    """

    def __init__(self, input_size, num_classes, dimensionality):
        super().__init__()

        self.num_classes = num_classes
        self.dimensionality = dimensionality

        if dimensionality == 2:
            self.logits = nn.Parameter(torch.ones((input_size[0], num_classes)))
            self.sum_dim = 1
        else:
            self.logits = nn.Parameter(torch.ones((*input_size, num_classes)))
            self.sum_dim = [1, 2, 3]

        self.logits.data.zero_()

    def logits_repeat(self, n):
        if self.dimensionality == 2:
            return self.logits.repeat(n, 1, 1)
        return self.logits.repeat(n, 1, 1, 1, 1)

    def log_prior(self, x):
        assert x.shape[1:] == self.logits.shape[:-1]
        x = nn.functional.one_hot(x, num_classes=self.num_classes).float()
        b = OneHotCategorical(logits=self.logits_repeat(x.shape[0]))
        log_p_nats = b.log_prob(x).sum(dim=self.sum_dim)
        return log_p_nats

    def sample_prior(self, N):
        data_distribution = Categorical(logits=self.logits)
        samples = data_distribution.sample([N])
        return samples


class CategoricalSplitPrior(nn.Module):
    """
    Factorized categorical splitprior with tunable logits.
    """

    def __init__(self, n_channels, args):
        super().__init__()

        self.num_classes = args.num_classes
        self.split_idx = n_channels - (n_channels // 2)
        self.nn = NN(args=args,
                     c_in=self.split_idx * self.num_classes,
                     c_out=(n_channels - self.split_idx) * self.num_classes,
                     nn_type=args.nn_type)

    def split(self, z):
        z1 = z[:, :self.split_idx]
        y = z[:, self.split_idx:]
        return z1, y

    def log_prior(self, x, logits):
        x = torch.nn.functional.one_hot(x, num_classes=self.num_classes).float()
        b = OneHotCategorical(logits=logits)
        log_p_nats = b.log_prob(x).sum(dim=[1, 2, 3])
        return log_p_nats

    def forward(self, z, reverse=False):
        if not reverse:
            z, y = self.split(z)
            py = self.nn(z).permute((0, 2, 3, 4, 1)).contiguous()
            return py, y, z
        else:
            py = self.nn(z).permute((0, 2, 3, 4, 1)).contiguous()
            y = self.sample_prior(py)
            return torch.cat([z, y], dim=1)

    def sample_prior(self, py):
        data_distribution = Categorical(logits=py)
        return data_distribution.sample()


def log_prior(x, logits, num_classes):
    x = torch.nn.functional.one_hot(x, num_classes=num_classes).float()
    b = OneHotCategorical(logits=logits)
    log_p_nats = b.log_prob(x).sum(dim=[1, 2, 3])
    return log_p_nats
