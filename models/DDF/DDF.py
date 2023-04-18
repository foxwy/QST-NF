import os
import numpy as np

import torch
from argparse import ArgumentParser

from training import Training

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))

parser = ArgumentParser()

parser.add_argument("--dataset", type=str, default="8gaussians", help="Training dataset [8gaussians/mnist/cityscapes]")
parser.add_argument("--nn_type", type=str, default="mlp", help="Type of NN in the coupling layers [mlp/densenet]")

parser.add_argument("--n_hidden_nn", type=int, default=256, help="Number of hidden units in coupling layers' NN")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
parser.add_argument("--net_epochs", type=int, default=100, help="Number of epochs to train the NN in coupling or splitprior layer")
parser.add_argument("--prior_epochs", type=int, default=30, help="Number of epochs to train the prior")
parser.add_argument("--k_sort", type=int, help="Parameter k in the discrete denoising coupling layer")


parser.add_argument("--with_splitprior", type=bool, default=False, help="Whether to train the model with splitpriors")
parser.add_argument("--save_model", type=bool, default=False, help="Whether to save the trained model")
parser.add_argument("--sample", type=bool, default=False, help="Whether to sample the model")

args = parser.parse_args()

if __name__ == "__main__":
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    K = 4
    povm = 'Pauli_4'

    with open('results.txt', 'w') as f:
        for N_p in [0, 0.5]:
            f.write("---%s\n" % N_p)
            f.flush()
            for N_q in [10, 20, 50, 80]:
                f.write("%s\n" % N_q)
                f.flush()

                N_s = N_q * 10**3
                if N_p == 0:
                    trainFileName = filepath + '/datasets/data/GHZ_MPS_' + povm + '_data_N' + str(N_q) + '.txt'
                else:
                    trainFileName = filepath + '/datasets/data/GHZ_MPS_' + povm + '_data_N_05_' + str(N_q) + '.txt'

                data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
                data_eval = np.loadtxt(trainFileName)[N_s:].astype(int)

                args.input_size = [N_q, 1, 1]
                args.num_classes = K
                args.num_building_blocks = 5
                args.n_hidden_nn = 64
                args.batch_size = 500
                args.lr = 0.001
                args.net_epochs = 10
                args.prior_epochs = 30
                args.nn_type = 'mlp'
                args.k_sort = K
                args.sample = True
                args.num_samples = 1

                training = Training(args, data_train, data_eval)
                training.train_2D(args, f)
