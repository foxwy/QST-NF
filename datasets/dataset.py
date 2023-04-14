# -*- coding: utf-8 -*-
# @Author: foxwy
# @Function: Provide some loss functions
# @Paper: Ultrafast quantum state tomography with feed-forward neural networks

import os
import sys
import numpy as np
import torch
import h5py

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

sys.path.append('..')
from datasets.data_generation import PaState
from Basis.Basic_Function import (array_posibility_unique, 
                                  data_combination, 
                                  num_to_groups)
from Basis.Basic_Function import (data_combination, 
                                  qmt, 
                                  qmt_pure, 
                                  qmt_torch, 
                                  qmt_torch_pure, 
                                  get_default_device)


def Dataset_P(rho_star, M, N, K, ty_state, p=1, seed=1):
    """
    Noise-free quantum measurements are sampled and some of these probabilities 
    are selected proportionally.

    Args:
        rho_star (tensor): The expected density matrix.
        M (tensor): The POVM, size (K, 2, 2).
        N (int): The number of qubits.
        K (int): The number of POVM elements.
        ty_state (str): The type of state, include 'mixed' and 'pure'.
        p (float): Selected measurement base ratio.
        seed (float): Random seed.

    Returns:
        tensor: Index of the sampled measurement base, with the zero removed.
        tensor: Probability distribution of sampling, with the zero removed.
        tensor: Probability distribution of sampling, include all measurement.
    """
    data_unique = data_combination(N, K, p, seed)  # part sample 

    if ty_state == 'pure':
        P = qmt_torch_pure(rho_star, [M] * N)
    else:
        P = qmt_torch(rho_star, [M] * N)
    P_idx = torch.arange(0, len(P), device=P.device)  # index of probability

    if p < 1:
        idxs = data_unique.dot(K**(np.arange(N - 1, -1, -1)))
        P = P[idxs]
        P_idx = P_idx[idxs]

    idx_nzero = P > 0
    return P_idx[idx_nzero], P[idx_nzero], P


def Dataset_sample(povm, state_name, N, sample_num, rho_p, ty_state, rho_star=0, M=None, read_data=False):
    """
    Quantum sampling with noise.

    Args:
        povm (str): The name of measurement, as Mea_basis().
        state_name (str): The name of state, as State().
        N (int): The number of qubits.
        sample_num (int): Number of samples to be sampled.
        rho_p (str): The P of Werner state, pure state when p == 1, identity matrix when p == 0.
        ty_state (str): The type of state, include 'mixed' and 'pure'.
        rho_star (array, tensor): The expect density matrix, assign the value directly if it exists, 
            otherwise regenerate it.
        M (tensor): The POVM, size (K, 2, 2).
        read_data (bool): If true, read the sample data from the ``.txt`` file.

    Returns:
        tensor: Index of the sampled measurement base, with the zero removed.
        tensor: Probability distribution of sampling, with the zero removed.
        tensor: Probability distribution of sampling, include all measurement.
    """
    if read_data:
        if 'P' in state_name:  # mix state
            trainFileName = filepath + '/datasets/data/' + state_name + \
                '_' + str(rho_p) + '_' + povm + '_data_N' + str(N) + '.txt'
        else:  # pure state
            trainFileName = filepath + '/datasets/data/' + \
                state_name + '_' + povm + '_data_N' + str(N) + '.txt'
        data_all = np.loadtxt(trainFileName)[:sample_num].astype(int)
    else:
        sampler = PaState(povm, N, state_name, rho_p, ty_state, M, rho_star) 
        #data_all, _ = sampler.samples_product(sample_num, save_flag=False)  # low
        P_idxs, P, P_all = sampler.sample_torch(sample_num, save_flag=False)  # faster

    #data_unique, P = array_posibility_unique(data_all)
    #return data_unique, P
    return P_idxs, P, P_all


def Dataset_sample_P(povm, state_name, N, K, sample_num, rho_p, ty_state, rho_star=0, read_data=False, p=1, seed=1):
    """The combination of ``Dataset_P`` and ``Dataset_sample``"""
    S_choose = data_combination(N, K, p, seed)
    S_choose_idxs = S_choose.dot(K**(np.arange(N - 1, -1, -1)))

    P_idxs, P, P_all = Dataset_sample(povm, state_name, N, sample_num, rho_p, ty_state, rho_star, read_data)
    P_choose = torch.zeros(len(S_choose), device=P_idxs.device)
    for i in range(len(P_choose)):
        if S_choose_idxs[i] not in P_idxs:
            P_choose[i] = 0
        else:
            P_choose[i] = P[torch.nonzero(P_idxs == S_choose_idxs[i])[0][0]]

    return S_choose_idxs, P_choose, P_all


def Dataset_RBM(N, K):  # RBM
    logfs = np.loadtxt('../RBM/logfs_'+str(N)+'.txt')
    P = np.exp(logfs)
    data_unique = data_combination(N, K)

    idxs = data_unique.dot(K**(np.arange(N - 1, -1, -1)))
    P_all = np.zeros(K**N)
    P_all[idxs] = P
    return idxs, P, P_all


def Dataset_RNN(povm, state_name, N, sample_num, rho_p, K, N_epoch=30, rho_star=0):  # RNN
    sampler = PaState(povm, N, state_name, rho_p, rho_star)
    _, data_train = sampler.samples_product(sample_num, save_flag=False)
    #samples_o, P_o = array_posibility_unique(data_origin)

    # generate samples
    N_sample = 10**5
    from models.RNN.rnn import LatentAttention
    model = LatentAttention(data_train=data_train, n_sample=sample_num, K=K, n_qubits=N, N_epoch=N_epoch, decoder='TimeDistributed_mol', Nsamples=N_sample)
    model.train()

    samples_r = np.loadtxt(filepath+'/models/GAN_MLE/samples/samples_' + str(N_epoch)+'.txt').astype(np.uint8)
    P_r = np.loadtxt(filepath+'/models/GAN_MLE/samples/P_'+str(N_epoch)+'.txt')

    # preprocessing
    '''
    data_unique = data_combination(N, K)
    P = np.zeros(len(data_unique))

    idxs_o = samples_o.dot(K**(np.arange(N - 1, -1, -1)))
    P[idxs_o] = np.sqrt(P_o)

    idxs_r = samples_r.dot(K**(np.arange(N - 1, -1, -1)))
    P[idxs_r] *= np.sqrt(P_r)

    return data_unique, P / sum(P)'''

    #return samples_o, P_o, samples_r, P_r
    idxs = samples_r.dot(K**(np.arange(N - 1, -1, -1)))
    P_all = np.zeros(K**N)
    P_all[idxs] = P_r
    return idxs, P_r, P_all


def Dataset_AQT(povm, state_name, N, sample_num, rho_p, K, N_epoch=50, rho_star=0):  # RNN
    sampler = PaState(povm, N, state_name, rho_p, rho_star)
    data_train, _ = sampler.samples_product(sample_num, save_flag=False)
    #samples_o, P_o = array_posibility_unique(data_train)

    Nl = 2
    dmodel = 64
    Nh = 4

    from models.Transformer.aqt import AQT
    samples, P = AQT(data_train=data_train, K=K, n_qubits=N, N_epoch=N_epoch, Nl=Nl, dmodel=dmodel, Nh=Nh)

    idxs = samples.dot(K**(np.arange(N - 1, -1, -1)))
    P_all = np.zeros(K**N)
    P_all[idxs] = P
    return idxs, P, P_all


def Dataset_DDM(povm, state_name, N, sample_num, rho_p, K, N_epoch=100, rho_star=0):  # RNN
    sampler = PaState(povm, N, state_name, rho_p, rho_star)
    data_train, _ = sampler.samples_product(sample_num, save_flag=False)
    #samples_o, P_o = array_posibility_unique(data_train)

    N_samples = 10**5

    from models.DDM.VariationalDiffusionModel_2 import VDM
    samples, P = VDM(data_train, N_samples, N_epoch, 1000, n_steps=1000)

    idxs = samples.dot(K**(np.arange(N - 1, -1, -1)))
    P_all = np.zeros(K**N)
    P_all[idxs] = P
    return idxs, P, P_all
