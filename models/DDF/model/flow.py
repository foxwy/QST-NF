import torch

from model.flow_layers import Squeeze, Permutation, Coupling
from model.categorical_prior import CategoricalSplitPrior


class Flow(torch.nn.Module):
    """
    Discrete Denoising Flow object; stores all the trained coupling and splitprior layers
    """

    def __init__(self, args):
        super().__init__()

        self.layers = []
        self.layers_ml = torch.nn.ModuleList(self.layers)

    def add_layer(self, layer):
        """
        Add layer to flow
        """
        assert any(isinstance(layer, c) for c in [Squeeze, CategoricalSplitPrior, Coupling, Permutation])
        self.layers.append(layer)
        self.layers_ml = torch.nn.ModuleList(self.layers)

    def forward(self, z, pys=(), ys=(), reverse=False):
        """
        Pass z through flow, forward or reverse
        """
        if not reverse:
            for layer in self.layers_ml:
                if isinstance(layer, (CategoricalSplitPrior)):
                    py, y, z = layer(z)
                    pys += (py,)
                    ys += (y,)
                else:
                    z = layer(z)
        else:
            for layer in reversed(list(self.layers_ml)):
                z = layer(z, reverse=True)
        return z, pys, ys
