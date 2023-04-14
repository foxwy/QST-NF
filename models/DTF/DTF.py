# -*- coding: utf-8 -*-
# @Author: yong
# @Date:   2023-03-25 22:49:51
# @Last Modified by:   WY
# @Last Modified time: 2023-04-14 13:26:43

import logging
import math
import DTF
import utilities 
import numpy as np
import time
import sys
from datetime import datetime
import pickle
import os
from argparse import ArgumentParser
import torch

filepath = os.path.abspath(os.path.join(os.getcwd(), '../..'))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

now = datetime.now()

np.set_printoptions(threshold=sys.maxsize)

# parser for cmd
parser = ArgumentParser()

# taskset -c 3 python test_dtf.py --exp=15 --num_TSPs=COPH --max_depth=3 --kfolds=1 --split_type=single_random
parser.add_argument("--exp", type=str, default='qdata', help="exp name COPH, COPM, COPW, 8Gaussian, MNIST, SNP, Mushroom")

parser.add_argument("--num_TSPs", type=int, default=10, help="Maximum number of TSPs")
parser.add_argument("--max_depth", type=int, default=3, help="Maximum Depth of a TSP")
parser.add_argument("--min_depth", type=int, default=1, help="min Depth of a TSP")

parser.add_argument("--SAMPLE", type=bool, default=True, help="Whether to sample new data")
parser.add_argument("--SAMPLE_AT_NODE", type=bool, default=False, help="Whether to sample at the node or not")
parser.add_argument("--CALCULATE_INV", type=bool, default=False, help="Whether to calculate inverse or not")
# next 2 not in use
parser.add_argument("--inc_depth", type=bool, default=False, help="Whether to increase the depth of the tree every 100 trees or not")
parser.add_argument("--every", type=int, default=100, help="increase depth every how many TSPs")

parser.add_argument("--kfolds", type=int, default=1, help="The number of kfolds to test")
parser.add_argument("--split_type", type=str, default="greedy_local_perm", help="The type of split to use, random vs GLP")
#random_multi or greedy_local_perm or single_random
parser.add_argument("--save_tree", type=bool, default=False, help="Whether to save the trained tree model")
parser.add_argument("--save_plots", type=bool, default=False, help="Whether to save the model outputs as plots")
parser.add_argument("--save_results", type=bool, default=False, help="Whether to save the results or not")

args = parser.parse_args()


if __name__ == '__main__':
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    K = 4
    povm = 'Pauli_4'

    with open('results.txt', 'w') as f:
        for N_p in [0]:
            f.write("---%s\n" % N_p)
            f.flush()
            for N_q in [10, 20]:
                f.write("%s\n" % N_q)
                f.flush()

                N_s = N_q * 10**3
                if N_p == 0:
                    trainFileName = filepath + '/datasets/data/GHZ_MPS_' + povm + '_data_N' + str(N_q) + '.txt'
                else:
                    trainFileName = filepath + '/datasets/data/GHZ_MPS_' + povm + '_data_N_05_' + str(N_q) + '.txt'

                data_train = np.loadtxt(trainFileName)[:N_s].astype(int)
                data_test = np.loadtxt(trainFileName)[N_s:].astype(int)

                args.min_samples_split = 200
                args.min_samples_leaf = 100

                args.result_dirs = os.path.join('results', now.strftime("%m%d%Y_%H%M%S"))
                args.output_filename = os.path.join(args.result_dirs, 'report_' + str(now.day) + '-' + str(now.month) + '-' + str(now.year) + '_' + str(now.hour) + '-' + str(now.minute) + '-' + str(now.second) + '.txt')

                # Create the Output directories
                if not os.path.exists(args.result_dirs):
                    os.makedirs(args.result_dirs)

                # Debugging flags
                DEBUG = True  # turn on debugging prints for this section
                PLOT = False

                # Other flags
                args.SAMPLE_AT_BEGIN = False  # Not used, leave it at False: To sample at the begining or not
                USE_pseudo_counts = False  # Not used, leave it at False:to use vs not to use pseudo-counts, not fully functional yet

                train_portion = 1  # to be used only when kfolds = 1
                alpha_smoothing = 0.001  # smoothing paramter in Q

                exp_results = []
                orig_probs = []

                n_samples, n_features, k = N_s, N_q, K

                # Max number of categories
                args.max_k = K  # np.amax(data_orig).astype(int) + 1  # to count categorey 0

                # Create the domain matrix of the root, domain matrices are dxk size
                args.domain_mat = np.full((n_features, args.max_k), np.nan)

                domain = []
                for i in range(0, n_features):
                    args.domain_mat[i, range(0, args.max_k)] = list(range(0, args.max_k))

                if(DEBUG):
                    print("k is: " + str(args.max_k) + " and that is the max k")
                #logger.info("Increasing depth is: "+str(args.inc_depth))

                start_depth = args.min_depth
                num_trees = args.num_TSPs

                for M in [args.max_depth]:  # this is one iteration

                    # write some info
                    utilities.write_to_report(args.output_filename, "########################################################################\n")
                    line = "For exp " + args.exp + " max depth is " + str(M) + " num_trees is " + str(args.num_TSPs) + " splitting strategy is " + args.split_type
                    utilities.write_to_report(args.output_filename, line + "\n")
                    logger.info(line)

                    current_dir = os.getcwd()
                    fold_filename = args.result_dirs + "/exp_" + args.exp + "_max_depth_" + str(M) + "_max_num_trees_" + str(args.num_TSPs) + "_splitstrat_" + args.split_type + "_samplenode_" + str(args.SAMPLE_AT_NODE) + ".obj"

                    # loop on kfolds
                    for i in range(0, args.kfolds):

                        total_train_time_temp = 0
                        per_tree_train_temp_Q_x = 0
                        total_test_time_temp = 0
                        total_params = 0
                        total_nodes = 0

                        # write some info
                        line = "*********FOR FOLD: " + str(i) + "*********"
                        utilities.write_to_report(args.output_filename, line + "\n")
                        logger.info(line)

                        # split train_data
                        train_data = data_train
                        test_data = data_test

                        # keep a copy of the original train_data to compare with the inverse for sanity check
                        orig_train_data = train_data.copy()
                        orig_test_data = test_data.copy()

                        logger.info("Size of training data is " + str(train_data.shape[0]))
                        logger.info("Size of testing data is " + str(test_data.shape[0]))

                        n, d = train_data.shape
                        n_test, d_test = test_data.shape

                        # if I am calculating the inverse store original train_data for sanity check later
                        if args.CALCULATE_INV:
                            permuted_train_datas = [orig_train_data]  # the first train_data is unpermuted
                            permuted_test_datas = [orig_test_data]  # the first test_data is unpermuted

                        # construct the trees
                        trees = []  # variable to keep the trees
                        tree_results = {"train_bt": [], "test_bt": [], "val_bt": [], "train": [], "test": [], "val": [], "train_t": [], "test_t": [], "val_t": [], "params": [], "num_nodes": []}  # the metrics, some of which will be empty.._bt is before training

                        # compute Q_x before training
                        start_time = time.time()
                        Q_x = utilities.compute_Q(train_data, args.domain_mat, args.max_k, alpha_smoothing)
                        end_time = time.time()
                        per_tree_train_temp_Q_x = end_time - start_time
                        total_train_time_temp += per_tree_train_temp_Q_x

                        # save NLLs before training
                        sum_samples_log_prob_train_bt, avg_samples_log_prob_train_bt = utilities.compute_avg_nll_all_Q(train_data, Q_x)
                        sum_samples_log_prob_test_bt, avg_samples_log_prob_test_bt = utilities.compute_avg_nll_all_Q(test_data, Q_x)
                        bpc_bt = sum_samples_log_prob_train_bt / (math.log(2) * n * d)
                        bpc_test_bt = sum_samples_log_prob_test_bt / (math.log(2) * n_test * d_test)

                        tree_results["train_bt"].append(avg_samples_log_prob_train_bt)
                        tree_results["test_bt"].append(avg_samples_log_prob_test_bt)

                        if args.inc_depth is True:
                            args.current_max_depth = args.min_depth
                        else:
                            args.current_max_depth = args.max_depth

                        for j in range(1, args.num_TSPs + 1):

                            line = "\nPlanting tree number " + str(j)
                            logger.info(line)

                            tree_filename = args.result_dirs + "/ver2_T_exp_" + args.exp + "_max_depth_" + str(M) + "_max_num_trees_" + str(args.num_TSPs) + "_current_tree_" + str(j) + "_fold_num_" + str(i) + "_splitstrat_" + args.split_type + "_pc_" + str(USE_pseudo_counts) + "_samplenode_" + str(args.SAMPLE_AT_NODE) + ".obj"
                            tree_results_filename = args.result_dirs + "/tree_res_ver2_T_exp_" + args.exp + "_max_depth_" + str(M) + "_max_num_trees_" + str(args.num_TSPs) + "_current_tree_" + str(j) + "_fold_num_" + str(i) + "_splitstrat_" + args.split_type + "_pc_" + str(USE_pseudo_counts) + "_samplenode_" + str(args.SAMPLE_AT_NODE) + ".obj"

                            utilities.write_to_report(args.output_filename, line + "\n")

                            if (args.inc_depth is True and j % args.every == 0 and args.current_max_depth < args.max_depth):
                                args.current_max_depth = args.current_max_depth + 1

                            # create tree number j
                            T, max_node_id, construction_time = DTF.construct_tree(train_data, args)

                            # save the tree
                            if args.save_tree:
                                tree_file = open(tree_filename, 'wb')
                                pickle.dump(T, tree_file)
                                tree_file.close()

                            # count the parameters
                            num_params, num_nodes = DTF.count_params(T)
                            total_params += num_params
                            total_nodes += num_nodes
                            tree_results["params"].append(total_params)
                            tree_results["num_nodes"].append(total_nodes)

                            start_time = time.time()

                            trees.append(T)
                            permuted_train_data = DTF.test_tree(trees[j - 1], train_data)  # Apply the permutations to the training data
                            Q_z = utilities.compute_Q(permuted_train_data, args.domain_mat, args.max_k, alpha_smoothing)  # compute Q of the permuted training data
                            end_time = time.time()

                            if (j == 1):  # for the first tree we add the time for computing Q_x
                                per_tree_train_time = per_tree_train_temp_Q_x + end_time - start_time + construction_time
                            else:
                                per_tree_train_time = end_time - start_time + construction_time

                            total_train_time_temp += per_tree_train_time

                            tree_results["train_t"].append(total_train_time_temp)

                            logger.info("Number of nodes in the tree is :" + str(max_node_id + 1))
                            line = "Training time for tree number " + str(j) + " is : " + str(total_train_time_temp)
                            logger.info(line)
                            utilities.write_to_report(args.output_filename, "For tree number " + str(j) + "\n")
                            utilities.write_to_report(args.output_filename, "Number of nodes in the tree is :" + str(max_node_id + 1) + "\n")
                            utilities.write_to_report(args.output_filename, line + "\n")

                            train_data = permuted_train_data

                            # Compute NLL of permuted train data
                            sum_samples_log_prob_train, avg_samples_log_prob_train = utilities.compute_avg_nll_all_Q(permuted_train_data, Q_z)

                            tree_results["train"].append(avg_samples_log_prob_train)
                            line = "The average NLL of the train train_data (before training) is " + str(avg_samples_log_prob_train_bt)
                            logger.info(line)
                            utilities.write_to_report(args.output_filename, line + "\n")

                            line = "The average NLL of the train train_data (after training) is " + str(avg_samples_log_prob_train)
                            logger.info(line)
                            utilities.write_to_report(args.output_filename, line + "\n")

                            if args.CALCULATE_INV:
                                permuted_train_datas.append(permuted_train_data.copy())

                            # Permute the test data
                            start_time = time.time()
                            permuted_test_data = DTF.test_tree(trees[j - 1], test_data)
                            end_time = time.time()
                            test_data = permuted_test_data
                            # test train_data is also modified after the call to test_tree
                            if args.CALCULATE_INV:
                                permuted_test_datas.append(permuted_test_data.copy())
                            # done with tes_data
                            test_time_per_tree = end_time - start_time

                            total_test_time_temp += test_time_per_tree
                            tree_results["test_t"].append(total_test_time_temp)

                            line = "Testing time for " + str(j) + " trees is " + str(total_test_time_temp)
                            logger.info(line)
                            utilities.write_to_report(args.output_filename, line + "\n")

                            # Calculate NLL of permuted test data
                            sum_samples_log_prob_test, avg_samples_log_prob_test = utilities.compute_avg_nll_all_Q(permuted_test_data, Q_z)

                            tree_results["test"].append(avg_samples_log_prob_test)
                            line = "The average NLL of the test data (before training) is " + str(avg_samples_log_prob_test_bt)
                            logger.info(line)

                            utilities.write_to_report(args.output_filename, line + "\n")
                            line = "The average NLL of the test data (after training) is " + str(avg_samples_log_prob_test)
                            f.write('nll test: {:.6f}\n'.format(avg_samples_log_prob_test))
                            f.flush()
                            logger.info(line)

                            utilities.write_to_report(args.output_filename, line + "\n")

                            if (j == args.num_TSPs and args.save_results):
                                tree_results_file = open(tree_results_filename, 'wb')
                                pickle.dump(tree_results, tree_results_file)
                                tree_results_file.close()

                        # End loop on number of trees
                        if args.SAMPLE:
                            times = []
                            for k in range(10):
                                torch.cuda.synchronize()
                                t0 = time.time()
                                samples_inv, p_inv = utilities.sample_invert_p(trees, Q_z, 1)
                                torch.cuda.synchronize()
                                t1 = time.time()
                                times.append(t1 - t0)
                            print('Times mean: {:.6f} std: {:.6f}\n'.format(np.mean(times), np.std(times)))
                            f.write('Times mean: {:.6f} std: {:.6f}\n\n'.format(np.mean(times), np.std(times)))
                            f.flush()

                        # calculate the inverse, just for sanity check
                        if args.CALCULATE_INV:
                            print("INVERSE CALCULATION")

                            for j in range(-1, -args.num_TSPs - 1, -1):
                                inv_data = DTF.calculate_inverse(trees[j], permuted_train_datas[j], permuted_train_datas[j - 1])
                                if DEBUG:
                                    print("Inverse for training train_data for tree " + str(num_trees + j) + " is calculated and is : ")
                                    print((inv_data == permuted_train_datas[j - 1]).all())
                                inv_test_data = DTF.calculate_inverse(trees[j], permuted_test_datas[j], permuted_test_datas[j - 1])
                                if DEBUG:
                                    print("Inverse for testing train_data for tree " + str(num_trees + j) + " is calculated and is : ")
                                    print((inv_test_data == permuted_test_datas[j - 1]).all())

                            print("The inverse for train data is calculated and is " + str((orig_train_data == inv_data).all()))  # just checking that inverse is correct
                            print("The inverse for test data  is calculated and is " + str((orig_test_data == inv_test_data).all()))  # just checking that inverse is correct
