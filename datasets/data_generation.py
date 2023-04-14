# -*- coding: utf-8 -*-
# @Author: foxwy
# @Function: Quantum state and quantum measurment
# @Paper: Ultrafast quantum state tomography with feed-forward neural networks

import sys
import time
import random
import numpy as np
from numpy.random import default_rng
import multiprocessing as mp
#from multiprocessing.pool import ThreadPool as Pool
import torch
#from torch.distributions.multinomial import Multinomial

sys.path.append('..')

# external libraries
from evaluation.ncon import ncon
from Basis.Basis_State import Mea_basis, MPS_state
from Basis.Basic_Function import (data_combination,
                                  data_combination_M2_single,
                                  qmt, 
                                  qmt_pure, 
                                  samples_mp, 
                                  qmt_torch, 
                                  qmt_torch_pure)


class PaState(Mea_basis):
    """
    Mimic quantum measurements to generate test samples.

    Examples::
        >>> sampler = PaState(basis='Tetra4', n_qubits=2, State_name='GHZi_P', P_state=0.4)
        >>> sampler = sample_torch(Ns=10000)
    """
    def __init__(self, basis='Tetra', n_qubits=2, State_name='GHZ', P_state=0.0, ty_state='mixed', M=None, rho_star=0):
        """
        Args:
            basis (str): The name of measurement, as Mea_basis().
            n_qubits (int): The number of qubits.
            State_name (str): The name of state, as State().
            P_state (float): The P of Werner state, pure state when p == 1, identity matrix when p == 0.
            ty_state (str): The type of state, include 'mixed' and 'pure'.
            M (array, tensor): The POVM.
            rho_star (array, tensor): The expect density matrix, assign the value directly if it exists, 
                otherwise regenerate it.
        """
        super().__init__(basis)
        self.N = n_qubits
        self.State_name = State_name
        self.p = P_state
        self.ty_state = ty_state

        if M is not None: # External assignment
            self.M = M
            
        if type(rho_star) is np.ndarray or type(rho_star) is torch.Tensor:  # External assignment
            self.rho = rho_star
        else:
            if self.ty_state == 'pure':
                self.rho, _ = self.Get_state_rho(State_name, n_qubits, P_state)
            else:
                _, self.rho = self.Get_state_rho(State_name, n_qubits, P_state)

    def samples_product(self, Ns=1000000, filename='N2', group_N=500, save_flag=True):
        """
        Faster using product structure and multiprocessing for batch processing.

        Args:
            Ns (int): The number of samples wanted.
            filename (str): The name of saved file.
            group_N (int): The number of samples a core can process at one time for collection, 
                [the proper value will speed up, too much will lead to memory explosion].
            save_flag (bool): If True, the sample data is saved as a file '.txt'.

        Returns:
            array: sample in k decimal.
            array: sample in onehot encoding.
        """
        if save_flag:
            if 'P' in self.State_name:  # mix state
                f_name = 'data/' + self.State_name + '_' + str(self.p) + '_' + self.basis + '_train_' + filename + '.txt'
                f2_name = 'data/' + self.State_name + '_' + str(self.p) + '_' + self.basis + '_data_' + filename + '.txt'
            else:  # pure state
                f_name = 'data/' + self.State_name + '_' + self.basis + '_train_' + filename + '.txt'
                f2_name = 'data/' + self.State_name + '_' + self.basis + '_data_' + filename + '.txt'

        if self.ty_state == 'pure':
            P_all = qmt_pure(self.rho, [self.M] * self.N)  # probs of all operators in product construction
        else:
            P_all = qmt(self.rho, [self.M] * self.N)  # probs of all operators in product construction

        # Multi-process sampling data
        if Ns < group_N:
            group_N = Ns

        params = [[P_all, group_N, self.K, self.N]] * (Ns // group_N)
        if Ns % group_N != 0:
            params.append([P_all, Ns % group_N, self.K, self.N])

        cpu_counts = mp.cpu_count()
        if len(params) < cpu_counts:
            cpu_counts = len(params)

        time_b = time.perf_counter()  # sample time
        print('---begin multiprocessing---')
        with mp.Pool(cpu_counts) as pool:
            results = pool.map(samples_mp, params)
            pool.close()
            pool.join()
        print('---end multiprocessing---')

        # Merge sampling results
        S_all = results[0][0]
        S_one_hot_all = results[0][1]
        print('num:', group_N)
        for num in range(1, len(results)):
            if Ns % group_N != 0 and num == len(results) - 1:
                print('num:', group_N * num + len(results[num][0]))
            else:
                print('num:', group_N * (num + 1))
            S_all = np.vstack((S_all, results[num][0]))
            S_one_hot_all = np.vstack((S_one_hot_all, results[num][1]))
        print('---finished generating samples---')

        if save_flag:
            print('---begin write data to text---')
            np.savetxt(f_name, S_one_hot_all, '%d')
            np.savetxt(f2_name, S_all, '%d')
            print('---end write data to text---')

        print('sample time:', time.perf_counter() - time_b)
        return S_all, S_one_hot_all

    def sample_torch(self, Ns=1000000, filename='N2', save_flag=True):
        """
        Sampling directly through [numpy multinomial] will be very fast.

        Args:
            Ns (int): The number of samples wanted.
            filename (str): The name of saved file.
            save_flag (bool): If True, the sample data is saved as a file '.txt'.

        Returns:
            tensor: Index of the sampled measurement base, with the zero removed.
            tensor: Probability distribution of sampling, with the zero removed.
            tensor: Probability distribution of sampling, include all measurement.
        """
        time_b = time.perf_counter()
        if self.ty_state == 'pure':
            P_all = qmt_torch_pure(self.rho, [self.M] * self.N)  # probs of all operators in product construction
        else:
            P_all = qmt_torch(self.rho, [self.M] * self.N)  # probs of all operators in product construction

        #counts = Multinomial(Ns, P_all).sample()
        rng = default_rng()
        P_all = P_all.cpu().numpy().astype(float)
        counts = rng.multinomial(Ns, P_all / sum(P_all))

        counts = torch.from_numpy(counts).to(self.rho.device)
        P_sample = (counts / Ns).to(torch.float32)
        P_idx = torch.arange(0, len(P_sample), device=P_sample.device)
        idx_use = P_sample > 0
        
        print('----sample time:', time.perf_counter() - time_b)

        return P_idx[idx_use], P_sample[idx_use], P_sample


class PaMPS(Mea_basis):
    def __init__(self, basis='Tetra', n_qubits=2, MPS_name='GHZ', p=0.0):
        super().__init__(basis)
        self.N = n_qubits
        self.MPS_name = MPS_name
        self.MPS = MPS_state(MPS_name, n_qubits).MPS

        # constructing the MPO for the locally depolarized GHZ state from its MPS representation
        USA = np.zeros((2, 2, 4, 4))

        E00 = np.zeros((4, 4))
        E10 = np.zeros((4, 4))
        E20 = np.zeros((4, 4))
        E30 = np.zeros((4, 4))
        E00[0, 0] = 1
        E10[1, 0] = 1
        E20[2, 0] = 1
        E30[3, 0] = 1

        USA = USA + np.sqrt(1.0 - p) * ncon((self.I, E00), ([-1, -2], [-3, -4]))
        USA = USA + np.sqrt(p / 3.0) * ncon((self.s1, E10), ([-1, -2], [-3, -4]))
        USA = USA + np.sqrt(p / 3.0) * ncon((self.s2, E20), ([-1, -2], [-3, -4]))
        USA = USA + np.sqrt(p / 3.0) * ncon((self.s3, E30), ([-1, -2], [-3, -4]))

        E0 = np.zeros((4))
        E0[0] = 1

        self.locMixer = ncon((USA, E0, np.conj(USA), E0), ([-1, -2, 1, 3], [3], [-4, -3, 1, 2], [2]))

        self.LocxM = ncon((self.M, self.locMixer), ([-3, 1, 2], [2, -2, -1, 1]))

        Tr = np.ones((self.K))

        self.l_P = [None] * self.N
        self.l_P[self.N - 1] = ncon((self.M, self.MPS[self.N - 1], self.MPS[self.N - 1],
                                     self.locMixer), ([-3, 1, 2], [3, -1], [4, -2], [2, 4, 3, 1]))

        for i in range(self.N - 2, 0, -1):
            self.l_P[i] = ncon((self.M, self.MPS[i], self.MPS[i], self.locMixer, self.l_P[i + 1], Tr),
                               ([-3, 4, 5], [-1, 6, 2], [-2, 7, 3], [5, 7, 6, 4], [2, 3, 1], [1]))

        self.l_P[0] = ncon((self.M, self.MPS[0], self.MPS[0], self.locMixer, self.l_P[1], Tr),
                           ([-1, 4, 5], [6, 2], [7, 3], [5, 7, 6, 4], [2, 3, 1], [1]))
        # print(self.l_P[0])

    def samples(self, Ns=1000000, filename='N2'):

        f = open('data/' + self.MPS_name + '_MPS_' + self.basis + '_train_' + filename + '.txt', 'w')
        f2 = open('data/' + self.MPS_name + '_MPS_' + self.basis + '_data_' + filename + '.txt', 'w')

        state = np.zeros((self.N), dtype=np.uint8)

        for ii in range(Ns):
            Pi = np.real(self.l_P[0])  # ncon((self.TrP[1],self.l_P[0]),([1,2],[1,2,-1]));
            Pnum = Pi

            i = 1

            # state[0] = np.random.choice(self.K, 1, p=Pi) #andsample(1:K,1,true,Pi);
            state[0] = np.argmax(np.random.multinomial(n=1, pvals=abs(Pi) / sum(abs(Pi)), size=1))

            Pden = Pnum[state[0]]

            PP = ncon((self.M[state[0]], self.locMixer, self.MPS[0], self.MPS[0]),
                      ([2, 1], [1, 4, 3, 2], [3, -1], [4, -2]))

            for i in range(1, self.N - 1):  # =2:N-1
                Pnum = np.real(ncon((PP, self.l_P[i]), ([1, 2], [1, 2, -1])))
                #print('pnum:', Pnum)
                Pi = Pnum / Pden
                # np.random.choice(self.K, 1, p=Pi) #randsample(1:K,1,true,Pi);
                state[i] = np.argmax(np.random.multinomial(n=1, pvals=abs(Pi) / sum(abs(Pi)), size=1))
                Pden = Pnum[state[i]]
                PP = ncon((PP, self.LocxM[:, :, state[i]], self.MPS[i], self.MPS[i]),
                          ([1, 2], [3, 4], [1, 3, -1], [2, 4, -2]));

            i = self.N - 1
            Pnum = np.real(ncon((PP, self.l_P[self.N - 1]), ([1, 2], [1, 2, -1])))
            Pi = Pnum / Pden

            # np.random.choice(self.K, 1, p=Pi) #randsample(1:K,1,true,Pi);
            #print(Pi)
            state[self.N - 1] = np.argmax(np.random.multinomial(n=1, pvals=abs(Pi) / sum(abs(Pi)), size=1))

            one_hot = np.squeeze(np.reshape(np.eye(self.K)[state], [1, self.N * self.K]).astype(np.uint8)).tolist()
            print(ii, state)
            # print one_hot
            for item in one_hot:
                f.write("%s " % item)

            f.write('\n')
            f.flush()

            for item in state:
                f2.write("%s " % item)

            f2.write('\n')
            f2.flush()

        f.close()
        f2.close()


#--------------------test--------------------
def Para_input():  # python data_generation.py Tetra 4 GHZ 0 1000
    print("basis", sys.argv[1])
    print("Number_qubits", int(sys.argv[2]))
    print("MPS", sys.argv[3])
    print("noise p ", float(sys.argv[4]))
    print("Nsamples", int(sys.argv[5]))
    sampler = PaState(basis=sys.argv[1], n_qubits=int(sys.argv[2]), MPS_name=sys.argv[3], p=float(sys.argv[4]))
    sampler.samples(Ns=int(sys.argv[5]))


#--------------------main--------------------
if __name__ == '__main__':
    for n_qubits in [10, 20, 50]:
        sampler = PaMPS(basis='Pauli_4', n_qubits=n_qubits)
        sampler.samples(Ns=n_qubits * 2 * 10**3, filename='N'+str(n_qubits))
    
    '''
    for i in [8]:
        print('-'*20, i)
        num_qubits = i
        sample_num = 10**6
        sampler = PaState(basis='Tetra4', n_qubits=num_qubits, State_name='W_P', P_state=1.0)
        sampler.samples_product(sample_num, 'N_t'+str(num_qubits), save_flag=True)'''

    '''
    num_qubits = 8
    sample_num = 100000
    sampler = PaState(basis='Tetra4', n_qubits=num_qubits, State_name='GHZi_P', P_state=0.4)

    t11 = time.perf_counter()
    sampler.samples_product(sample_num, 'N'+str(num_qubits), save_flag=False)
    t21 = time.perf_counter()

    print('sampe time:', t21 - t11)'''

    '''
    B = Mea_basis('Tetra4')
    #rho = (B.I + 0.5*B.X + np.sqrt(3)/2*B.Y)/2
    s, rho = B.Get_state_rho('W', 10)
    #print('rho:', rho)
    #print(B.M)
    t1 = time.perf_counter()
    P = qmt(rho, [B.M]*10)
    t2 = time.perf_counter()
    print(t2-t1)
    print(P, sum(P))
    #B.Basis_info()'''
