import os
import time
import numpy as np
import torch

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))

import ann as A


# Basic parameters
def AQT(data_train, data_test, K, n_qubits, N_epoch, Nl=2, dmodel=64, Nh=4):
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    Nq = n_qubits
    Nep = N_epoch

    # Load data
    data = data_train
    np.random.shuffle(data)

    # Train model
    model = A.InitializeModel(
        Nq, Nlayer=Nl, dmodel=dmodel, Nh=Nh, Na=K, device=device).to(device)

    t = time.time()
    model, loss = A.TrainModel(
        model, data, device, batch_size=500, lr=1e-3, Nep=Nep)
    print('Took %f minutes' % ((time.time() - t) / 60))

    # sample time
    times = []
    for k in range(10):
        torch.cuda.synchronize()
        t0 = time.time()
        model.samples(1)
        torch.cuda.synchronize()
        t1 = time.time()
        times.append(t1 - t0)
    print('Times mean: {:.6f}  std: {:.6f}'.format(np.mean(times), np.std(times)))

    # nll
    samples = data_test
    pt_model = model.p(samples)
    nll_test = -np.mean(np.log(pt_model))
    print('nll_test:', nll_test)

    return times, nll_test


if __name__ == '__main__':
    K = 4
    N_epoch = 50
    povm = 'Pauli_4'

    with open('results.txt', 'w') as f:
        for N_p in [0, 0.5]:
            f.write("---%s\n" % N_p)
            f.flush()
            for N_q in [10, 20, 50, 80, 100]:
                f.write("%s\n" % N_q)
                f.flush()

                N_s = N_q * 10**3
                if N_p == 0:
                    trainFileName = filepath + '/datasets/data/GHZ_MPS_' + povm + '_data_N' + str(N_q) + '.txt'
                else:
                    trainFileName = filepath + '/datasets/data/GHZ_MPS_' + povm + '_data_N_05_' + str(N_q) + '.txt'

                data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
                data_test = np.loadtxt(trainFileName)[N_s:].astype(int)

                times, nll_test = AQT(data_train=data_train, data_test=data_test, K=K, n_qubits=N_q, N_epoch=N_epoch)
                f.write('Times mean: {:.6f} std: {:.6f}\n'.format(np.mean(times), np.std(times)))
                f.write('nll test: {:.6f}\n\n'.format(nll_test))
                f.flush()
