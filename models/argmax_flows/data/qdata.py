import os
import random
import numpy as np
import torch
import torch.utils.data as data


def add_data_args(parser):
    parser.add_argument('--batch_size', type=int, default=200)


class QData(data.Dataset):
    def __init__(self, vocab_size, dataset):
        # create data
        self.tensors = torch.from_numpy(dataset).long()
        self.indv_samples = torch.from_numpy(np.unique(self.tensors, axis=0)).float()
        self.length = len(dataset)
        self.vocab_size = vocab_size

    def __getitem__(self, index):
        idx = random.randint(0, self.tensors.shape[0] - 1)
        x = self.tensors[idx]
        # x = self.tensors[index]
        return x

    def __len__(self):
        return self.length


def get_qdata(num_classes, batch_size, dataset):
    train_dataset = QData(num_classes, dataset)
    trainloader = data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    return trainloader


def get_GHZ_data(povm, N_q, N_s):
    filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
    trainFileName = filepath + '/datasets/data/GHZ_MPS_'+povm+'_data_N' + str(N_q) + '.txt'
    data_train = np.loadtxt(trainFileName)[:N_s].astype(int)

    return data_train


def int2bin(num_array):  # numpy
    (N, L) = num_array.shape
    b = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    return np.reshape(b[num_array], [N, L*2])


def bin2int(num_array):
    (N, L) = num_array.shape
    a = np.reshape(num_array, [N, int(L / 2), 2])
    b = np.array([2, 1])
    return a.dot(b)


def cycle(dl):
    while True:
        for data in dl:
            yield data


if __name__ == '__main__':
    print(os.path.abspath(os.path.join(os.getcwd(), '../..')))
