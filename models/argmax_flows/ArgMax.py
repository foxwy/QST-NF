# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2023-04-12 09:27:09
# @Last Modified by:   WY
# @Last Modified time: 2023-04-14 13:27:05

import os
import numpy as np
import argparse

from experiment.flow import FlowExperiment, add_exp_args
from model.model import get_model, get_model_id, add_model_args
from optim.expdecay import get_optim, get_optim_id, add_optim_args
from data.qdata import add_data_args, get_qdata

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))

parser = argparse.ArgumentParser()

add_exp_args(parser)
add_data_args(parser)
add_model_args(parser)
add_optim_args(parser)
args = parser.parse_args()


if __name__ == "__main__":
    print("Args:" + str(args))

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

                train_loader = get_qdata(K, args.batch_size, data_train.reshape(len(data_train), 1, N_q))
                eval_loader = get_qdata(K, args.batch_size, data_eval.reshape(len(data_eval), 1, N_q))
                data_shape = (1, N_q)
                num_classes = K

                model = get_model(args, data_shape=data_shape, num_classes=num_classes)
                model_id = get_model_id(args)

                optimizer, scheduler_iter, scheduler_epoch = get_optim(args, model)
                optim_id = get_optim_id(args)

                exp = FlowExperiment(args=args,
                                     model_id=model_id,
                                     optim_id=optim_id,
                                     train_loader=train_loader,
                                     eval_loader=eval_loader,
                                     model=model,   
                                     optimizer=optimizer,
                                     scheduler_iter=scheduler_iter,
                                     scheduler_epoch=scheduler_epoch,
                                     f=f)

                exp.run()