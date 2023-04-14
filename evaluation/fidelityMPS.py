# --------------------libraries--------------------
# external libraries
import numpy as np
import torch
#import tensorly as tl
#from tensorly.decomposition import matrix_product_state
from scipy.linalg import sqrtm
import matplotlib.pyplot as plt
import multiprocessing as mp

import sys
sys.path.append('..')

# internal libraries
from Basis.Basis_State import Mea_basis, MPS_state
from evaluation.ncon import ncon


# --------------------class--------------------
# -----Fid_MPS-----
class Fid_MPS(Mea_basis):
    def __init__(self, basis='Tetra', Number_qubits=2, MPS='GHZ', Nmax=60000):
        super().__init__(basis)
        self.N = Number_qubits
        self.Nmax = Nmax

        # Tensor for expectation value
        self.Trsx = np.zeros((self.N, self.K), dtype=complex)
        self.Trsy = np.zeros((self.N, self.K), dtype=complex)
        self.Trsz = np.zeros((self.N, self.K), dtype=complex)
        self.Trrho = np.zeros((self.N, self.K), dtype=complex)
        self.Trrho2 = np.zeros((self.N, self.K, self.K), dtype=complex)
        self.T2 = np.zeros((self.N, self.K, self.K), dtype=complex)

        # MPS state
        self.MPS = MPS_state(MPS, Number_qubits).MPS

    # ----------fidelity----------
    def Fidelity(self, S):  # S(sample):[[0, 3, 2, 1], [1, 0, 3, 1]]
        Fidelity = 0.0
        #F2 = 0.0
        Ns = S.shape[0]

        for i in range(Ns):
            # contracting the entire TN for each sample S[i,:]
            eT = ncon((self.it[:, S[i, 0]], self.M, self.MPS[0],
                       self.MPS[0]), ([3], [3, 2, 1], [1, -1], [2, -2]))

            for j in range(1, self.N-1):
                eT = ncon((eT, self.it[:, S[i, j]], self.M, self.MPS[j], self.MPS[j]), ([
                          2, 4], [1], [1, 5, 3], [2, 3, -1], [4, 5, -2]))

            j = self.N - 1
            eT = ncon((eT, self.it[:, S[i, j]], self.M, self.MPS[j], self.MPS[j]), ([
                      2, 5], [1], [1, 4, 3], [3, 2], [4, 5]))

            Fidelity = Fidelity + eT

        Fidelity = np.abs(Fidelity / float(Ns))

        return np.real(Fidelity)

    def cf(self, params):
        S = params[0]
        P_s = params[1]
        Fidelity = 0.0

        for i in range(len(S)):
            '''
            P = ncon((self.MPS[0], self.MPS[0], self.M[S[i, 0], :, :]), ([1, -1], [2, -2], [1, 2]))

            # contracting the entire TN for each sample S[i,:]
            for j in range(1, self.N - 1):
                P = ncon((P, self.MPS[j], self.MPS[j], self.M[S[i, j], :, :]), ([1, 2], [1, 3, -1], [2, 4, -2], [3, 4]))

            P = ncon((P, self.MPS[self.N - 1], self.MPS[self.N - 1], self.M[S[i, self.N - 1], :, :]), ([1, 2], [3, 1], [4, 2], [3, 4]))

            #print(P.real, P_s[i])
            if P_s[i] > 1e-24:
                Fidelity += np.sqrt(P.real / P_s[i])'''
            Fidelity += np.log(P_s[i])

        return Fidelity

    def cFidelity(self, S, P_s, b_s=10**4):  # S(real):[[0, 3, 2, 1], [1, 0, 3, 1]], logP(sample):possibility 采样计算方法
        Fidelity = 0.0
        Ns = S.shape[0]
        #Ns = clamp(Ns, 0, self.Nmax)

        Ncalls = int(Ns / b_s)
        params = []
        for k in range(Ncalls):
            params.append([S[k * b_s:(k + 1) * b_s, :], P_s[k * b_s:(k + 1) * b_s]])

        cpu_counts = mp.cpu_count()
        if Ncalls < cpu_counts:
            cpu_counts = Ncalls
        with mp.Pool(cpu_counts) as pool:
            results = pool.map(self.cf, params)

        Fidelity = np.abs(sum(results) / float(Ns))

        return Fidelity

    def cFidelity2(self, S, Pt):
        Fidelity = 0.0
        Ns = S.shape[0]

        for i in range(Ns):
            P = ncon((self.MPS[0], self.MPS[0], self.M[S[i, 0], :, :]), ([
                     1, -1], [2, -2], [1, 2]))

            # contracting the entire TN for each sample S[i,:]
            for j in range(1, self.N-1):
                P = ncon((P, self.MPS[j], self.MPS[j], self.M[S[i, j], :, :]), ([
                         1, 2], [1, 3, -1], [2, 4, -2], [3, 4]))

            P = ncon((P, self.MPS[self.N-1], self.MPS[self.N-1],
                      self.M[S[i, self.N-1], :, :]), ([1, 2], [3, 1], [4, 2], [3, 4]))

            Fidelity += np.sqrt(P.real * Pt[i])

        return np.real(Fidelity)


# --------------------main--------------------
if __name__ == '__main__':
    fid = Fid_MPS()
    print(fid.basis)
    fid.Basis_info
    print(fid.MPS)
    GHZ_state, GHZ_rho = fid.Get_GHZ(3)
    fid.Rho_info(GHZ_rho)
