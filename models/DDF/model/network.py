import torch
import torch.nn as nn
import torch.nn.functional as F


class DenseLayer(nn.Module):
    """
    Single layer of DenseNet
    """

    def __init__(self, n_inputs, growth):
        super().__init__()

        self.nn = torch.nn.Sequential(
            nn.Conv2d(n_inputs, n_inputs, kernel_size=1, padding=0),
            nn.ReLU(),
            nn.Conv2d(n_inputs, growth, kernel_size=3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        h = self.nn(x)
        h = torch.cat([x, h], dim=1)
        return h


class DenseBlock(nn.Module):
    """
    Block of DenseNet, consisting of multiple DenseLayers
    """

    def __init__(
            self, depth, n_inputs, n_outputs):
        super().__init__()

        future_growth = n_outputs - n_inputs

        layers = []

        for d in range(depth):
            growth = future_growth // (depth - d)

            layers.append(DenseLayer(n_inputs, growth))
            n_inputs += growth
            future_growth -= growth

        self.nn = torch.nn.Sequential(*layers)

    def forward(self, x):
        return self.nn(x)


class NN(nn.Module):
    """
    Neural Network, either DenseNet or simple MLP
    """

    def __init__(
            self, args, c_in, c_out, nn_type):
        super().__init__()

        self.nn_type = nn_type
        self.num_classes = args.num_classes

        n_hidden = args.n_hidden_nn

        if nn_type == 'mlp':

            layers = [
                torch.nn.Linear(c_in, n_hidden),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.2),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.2),
                torch.nn.Linear(n_hidden, n_hidden),
                torch.nn.BatchNorm1d(n_hidden),
                torch.nn.ReLU(),
                #torch.nn.Dropout(0.2),
                torch.nn.Linear(n_hidden, c_out)
            ]

        elif nn_type == 'densenet':

            layers = [DenseBlock(depth=args.densenet_depth, n_inputs=c_in, n_outputs=n_hidden + c_in),
                      nn.Conv2d(n_hidden + c_in, c_out, kernel_size=3, padding=1)]

        else:
            raise ValueError

        self.nn = torch.nn.Sequential(*layers)

        # Set parameters of last conv-layer to zero.
        self.nn[-1].weight.data.zero_()
        self.nn[-1].bias.data.zero_()

        self.to(args.device)

    def forward(self, x1):

        if self.nn_type == 'mlp':
            x1 = F.one_hot(x1, num_classes=self.num_classes).float().view(len(x1), -1)
            out = self.nn(x1)
            return out.view(len(out), self.num_classes, -1)

        # densenet: x is of shape [bs, n_channels, height, width] with values in {0, .., self.num_classes}
        x1 = F.one_hot(x1, num_classes=self.num_classes).float()
        x1 = x1.view(x1.shape[0], x1.shape[1] * x1.shape[4], x1.shape[2], x1.shape[3])
        out = self.nn(x1)
        out = out.view(out.shape[0], self.num_classes, out.shape[1] // self.num_classes, out.shape[2], out.shape[3])
        return out
