import torch


class Model(torch.nn.Module):
    """
    This object stores the trained flow and prior
    """

    def __init__(self, flow, prior):
        super().__init__()

        self.flow = flow
        self.prior = prior

    def forward(self, x):
        """
        Decode from data space to prior space
        """
        z, _, _ = self.flow(x)
        log_pz = self.prior.log_prior(z)
        return -torch.mean(log_pz)

    def inverse(self, z):
        """
        Decode from prior space to data space
        """
        return self.flow(z, reverse=True)

    def sample(self, num_samples):
        """
        Sample from input distribution by sampling from prior distribution and applying flow
        """
        z_sample = self.prior.sample_prior(num_samples)
        log_pz = self.prior.log_prior(z_sample)
        out, _, _ = self.inverse(z_sample)
        return out, log_pz
