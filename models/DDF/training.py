import logging
import os.path
import sys
import time

import torch
import numpy as np

from data.qdata import get_qdata
from model.categorical_prior import CategoricalPrior, CategoricalSplitPrior, log_prior
from model.flow import Flow
from model.flow_layers import Squeeze, Permutation, Coupling
from model.model import Model
from model.network import NN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Training:
    """
    Handles the training loop.
    """

    def __init__(self, args, dataset=None, data_eval=None):

        self.dataset = args.dataset
        self.num_classes = args.num_classes
        self.k_sort = args.k_sort

        self.save_model = args.save_model
        self.sample = args.sample
        self.num_samples = args.num_samples

        self.device = args.device

        self.input_size = args.input_size
        self.n_channels, self.height, self.width = args.input_size  # this will be updated

        self.flow = Flow(args).to(args.device)  # start with empty flow

        self.train_loader = get_qdata(args.num_classes, args.batch_size, dataset)
        self.eval_loader = get_qdata(args.num_classes, args.batch_size, data_eval)
        self.dimensionality = 2

    def train_splitprior(self, args):
        """
        Train one splitprior layer network.
        """
        splitprior = CategoricalSplitPrior(self.n_channels, args)
        optimizer = torch.optim.Adam(splitprior.parameters(), lr=args.lr)

        for epoch in range(args.net_epochs):

            running_loss = 0.0

            for data_batch in self.train_loader:

                data_batch, _, _ = self.flow(data_batch.to(args.device))

                py, y, z = splitprior(data_batch)
                log_py = splitprior.log_prior(y, py)

                loss = - log_py.mean()
                bpd = loss / np.prod(self.input_size) / np.log(2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += bpd.item()

            logger.info('[%d] Loss: %.3f' % (epoch + 1, running_loss / len(self.train_loader)))
            running_loss = 0.0

        return splitprior

    def train_net(self, args):
        """
        Train one coupling layer network
        """
        split_idx = self.n_channels - (self.n_channels // 2)
        net = NN(args,
                 c_in=split_idx * self.num_classes,
                 c_out=(self.n_channels - split_idx) * self.num_classes,
                 nn_type=args.nn_type)

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)

        for epoch in range(args.net_epochs):

            running_loss = 0.0
            for data_batch in self.train_loader:

                x, _, _ = self.flow(data_batch.to(args.device))

                x1 = x[:, :split_idx]
                x2 = x[:, split_idx:]

                p_x2_given_x1 = net(x1)

                loss = criterion(p_x2_given_x1, x2.long())

                # backward + update
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            logger.info('[%d] Loss: %.3f' % (epoch + 1, running_loss / len(self.train_loader)))
            running_loss = 0.0

        logger.info('#' * 20 + '\n')
        return net

    def train_prior(self, args):
        """
        Train categorical prior distribution
        """
        logger.info("train prior")
        prior = CategoricalPrior([self.n_channels, self.height, self.width], self.num_classes, self.dimensionality).to(
            args.device)

        optimizer = torch.optim.Adam(prior.parameters(), lr=.01)

        for epoch in range(args.prior_epochs):

            running_loss = 0.0
            loss_ll = 0.0
            for data_batch in self.train_loader:

                z, pys, ys = self.flow(data_batch.to(args.device))

                log_pz = prior.log_prior(z)

                for py, y in zip(pys, ys):
                    log_py = log_prior(y, py, self.num_classes)
                    log_pz += log_py

                loss = -torch.mean(log_pz)
                bpd = loss / np.prod(self.input_size) / np.log(2)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += bpd.item()
                loss_ll += loss.item()

            logger.info('[%d] Loss: %.3f BPD %.3f NLL' % (epoch + 1, running_loss / len(self.train_loader), loss_ll / len(self.train_loader)))
            running_loss = 0.0

        logger.info('#' * 20 + '\n')
        return prior

    def add_coupling_layer(self, args):
        """
        Train a new coupling layer on top of the current flow and then add it to the flow
        """
        # add permutation
        logger.info("add coupling")
        self.flow.add_layer(Permutation(self.n_channels))
        net = self.train_net(args)
        self.flow.add_layer(Coupling(self.num_classes, self.k_sort, self.n_channels, net, self.dimensionality))

    def add_squeeze_layer(self):
        """
        Add a squeeze layer to the flow
        """
        logger.info("add squeeze")
        self.flow.add_layer(Squeeze())
        self.n_channels *= 4
        self.height //= 2
        self.width //= 2

    def add_splitprior_layer(self, args):
        """
        Train a new splitprior layer on top of the current flow and then add it to the flow
        """
        logger.info("add splitprior")
        splitprior = self.train_splitprior(args)
        self.flow.add_layer(splitprior)
        self.n_channels //= 2

    def evaluate(self, prior, args, f):
        """
        Evaluate the model (i.e. flow + prior) on the test data set
        """
        logger.info("evaluate on evaluation set")
        with torch.no_grad():

            self.flow.eval()
            prior.eval()

            loss_ll = 0.0

            for data_batch in self.eval_loader:

                z, pys, ys = self.flow(data_batch.to(args.device))

                log_pz = prior.log_prior(z)

                for py, y in zip(pys, ys):
                    log_py = log_prior(y, py, self.num_classes)
                    log_pz += log_py

                loss = -torch.mean(log_pz)

                loss_ll += loss.item()

            logger.info('EVALUATION: %.3f NLL' % (loss_ll / len(self.eval_loader)))

        return loss_ll / len(self.eval_loader)

    def train_2D(self, args, f):
        """
        Train a single coupling layer on two dimensional toy data
        """
        for i in range(args.num_building_blocks):
            # Coupling
            self.add_coupling_layer(args)
            prior = self.train_prior(args)  # train prior to get current BPD
            nll_test = self.evaluate(prior, args, f)

            model = Model(self.flow, prior)

            if self.sample:
                with torch.no_grad():
                    times = []
                    for k in range(10):
                        torch.cuda.synchronize()
                        t0 = time.time()
                        samples, log_pz = model.sample(self.num_samples)
                        torch.cuda.synchronize()
                        t1 = time.time()
                        times.append(t1 - t0)
                    print('Times mean: {:.6f} std: {:.6f}\n'.format(np.mean(times), np.std(times)))

                    f.write('Times mean: {:.6f} std: {:.6f}\n'.format(np.mean(times), np.std(times)))
            f.write('nll test: {:.6f}\n\n'.format(nll_test))
            f.flush()
