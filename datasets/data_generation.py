# -*- coding: utf-8 -*-
# @Author: foxwy
# @Function: Quantum state and quantum measurment
# @Paper: Ultrafast quantum state tomography with feed-forward neural networks

import sys
import argparse
import numpy as np

sys.path.append('..')

# external libraries
from evaluation.ncon import ncon
from Basis.Basis_State import Mea_basis, MPS_state


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


#--------------------main--------------------
if __name__ == '__main__':
    print('-'*20+'set parser'+'-'*20)
    parser = argparse.ArgumentParser()
    parser.add_argument("--p", type=float, default=1, help="p of Werner state")
    args = parser.parse_args()

    for n_qubits in [10, 20, 50, 80]:
        sampler = PaMPS(basis='Pauli_4', n_qubits=n_qubits, p=1-args.p)
        sampler.samples(Ns=n_qubits * 2 * 10**3, filename='N'+str(n_qubits))
