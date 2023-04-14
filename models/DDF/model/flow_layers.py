import torch
from torch.nn.functional import softmax


class Coupling(torch.nn.Module):
    """
    Denoising coupling layer
    """

    def __init__(self, num_classes, k_sort, n_channels, NN, dimensionality):
        super().__init__()

        self.split_idx = n_channels - (n_channels // 2)
        self.NN = NN
        self.NN.eval()

        self.dimensionality = dimensionality
        self.num_classes = num_classes
        self.k_sort = k_sort

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def conditional_permutation(self, p_x2_given_x1, x2, reverse):
        """
        Performs the conditional permutation operation
        """
        assert p_x2_given_x1.shape == x2.shape
        assert p_x2_given_x1.shape[-1] == self.num_classes

        p_x2_given_x1 = p_x2_given_x1.view(-1, self.num_classes)
        x2 = x2.view(-1, self.num_classes)

        perm = torch.arange(self.num_classes).repeat(p_x2_given_x1.shape[0], 1).to(self.device)

        for i in range(self.k_sort):
            max_idx = torch.argmax(p_x2_given_x1[:, i:], dim=1) + i
            p_x2_given_x1.scatter_(dim=1, index=max_idx.unsqueeze(1), src=p_x2_given_x1[:, i].unsqueeze(1))
            p_i = perm.gather(dim=1, index=max_idx.unsqueeze(1)).squeeze()
            perm.scatter_(dim=1, index=max_idx.unsqueeze(1), src=perm[:, i].unsqueeze(1))
            perm[:, i] = p_i

        if reverse:
            ranges = torch.arange(self.num_classes).repeat(perm.shape[0], 1).to(self.device)
            perm = torch.empty_like(perm).scatter_(dim=1, index=perm, src=ranges).to(self.device)

        y2 = torch.gather(x2, dim=1, index=perm)

        return y2

    def forward(self, x, reverse=False):

        x1 = x[:, :self.split_idx]
        x2 = x[:, self.split_idx:]

        # get network prediction of x2 based on x1
        p_x2_given_x1 = self.NN(x1)

        if self.dimensionality == 2:
            p_x2_given_x1 = p_x2_given_x1.permute((0, 2, 1)).contiguous()
        else:
            p_x2_given_x1 = p_x2_given_x1.permute((0, 2, 3, 4, 1)).contiguous()

        x2_oh = torch.nn.functional.one_hot(x2, num_classes=self.num_classes).float()

        y2 = self.conditional_permutation(p_x2_given_x1=p_x2_given_x1,
                                          x2=x2_oh,
                                          reverse=reverse)
        y2 = y2.view(x2_oh.shape).argmax(dim=-1)

        return torch.cat([x1, y2], dim=1)


class Permutation(torch.nn.Module):
    """
    Channel-wise permutation layer
    """

    def __init__(self, size):
        super().__init__()

        self.p = torch.randperm(size)
        self.p_inv = torch.zeros(size, dtype=int)
        self.p_inv[self.p] = torch.arange(size)

    def forward(self, z, reverse=False):
        if not reverse:
            return z[:, self.p]
        return z[:, self.p_inv]


class Squeeze(torch.nn.Module):
    """
    Squeeze layer
    Contains code from https://github.com/jornpeters/integer_discrete_flows/blob/master/models/generative_flows.py .
    """

    def __init__(self):
        super().__init__()

    def space_to_depth(self, x):
        xs = x.size()
        # Pick off every second element
        x = x.view(xs[0], xs[1], xs[2] // 2, 2, xs[3] // 2, 2)
        # Transpose picked elements next to channels.
        x = x.permute((0, 1, 3, 5, 2, 4)).contiguous()
        # Combine with channels.
        x = x.view(xs[0], xs[1] * 4, xs[2] // 2, xs[3] // 2)
        return x

    def depth_to_space(self, x):
        xs = x.size()
        # Pick off elements from channels
        x = x.view(xs[0], xs[1] // 4, 2, 2, xs[2], xs[3])
        # Transpose picked elements next to HW dimensions.
        x = x.permute((0, 1, 4, 2, 5, 3)).contiguous()
        # Combine with HW dimensions.
        x = x.view(xs[0], xs[1] // 4, xs[2] * 2, xs[3] * 2)
        return x

    def forward(self, z, reverse=False):
        if not reverse:
            return self.space_to_depth(z)
        return self.depth_to_space(z)
