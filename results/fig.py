# -*- coding: utf-8 -*-
# @Author: WY
# @Date:   2022-10-14 09:17:52
# @Last Modified by:   yong
# @Last Modified time: 2023-04-18 22:17:41

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import ConnectionPatch
import scienceplots

plt.style.use(['ieee'])
plt.rcParams["font.family"] = 'Times New Roman'

font_size = 24  # 34, 28, 38
font = {'size': font_size, 'weight': 'normal'}


def Plt_set(ax, xlabel, ylabel, savepath, log_flag=0, loc=4):
    ax.legend(loc=loc)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if log_flag == 1:
        ax.set_xscale('log')
    if log_flag == 2:
        ax.set_yscale('log')
    if log_flag == 3:
        ax.set_xscale('log')
        ax.set_yscale('log')

    plt.savefig(savepath + '.pdf', dpi=600, bbox_inches='tight', pad_inches=0.02)


if __name__ == '__main__':
    colors = ['#ff3efc', 'blue', '#4b9cc8', 'red', 'gray']
    style = ['-', '-', '--', '--', '--']
    N_q = [10, 20, 50, 80]
    b_path = '../models/'
    model_names = ['RNN', 'Transformer', 'argmax_flows', 'DDF', 'DTF']

    # NLL
    # p=1
    fig, ax = plt.subplots(1, 1)
    axins = ax.inset_axes((0.55, 0.1, 0.4, 0.35))
    for idx, model in enumerate(model_names):
        r_files = b_path + model + '/results.txt'

        nll = []
        with open(r_files, 'r') as f:
            data = f.readlines()[:16]
            for i in range(len(data)):
                if 'nll' in data[i]:
                    nll.append(float(data[i][10:19]))
            print(nll)

        ax.plot(N_q, nll, style[idx], label=model, color=colors[idx])
        axins.plot(N_q, nll, style[idx], label=model, color=colors[idx])
    ax.set_xticks(N_q)
    xlim0 = 45
    xlim1 = 55
    ylim0 = 46
    ylim1 = 58
    ax.plot([xlim0, xlim1, xlim1, xlim0, xlim0], [ylim0, ylim0, ylim1, ylim1, ylim0], '--', color='black', linewidth=1)
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    xy = (xlim1, ylim0)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    Plt_set(ax, 'Number of qubits', 'NLL', 'nll_1', loc=2)

    # p=0.5
    fig, ax = plt.subplots(1, 1)
    axins = ax.inset_axes((0.55, 0.1, 0.4, 0.35))
    for idx, model in enumerate(model_names):
        r_files = b_path + model + '/results.txt'

        nll = []
        with open(r_files, 'r') as f:
            data = f.readlines()[19:]
            for i in range(len(data)):
                if 'nll' in data[i]:
                    nll.append(float(data[i][10:19]))
            print(nll)

        ax.plot(N_q, nll, style[idx], label=model, color=colors[idx])
        axins.plot(N_q, nll, style[idx], label=model, color=colors[idx])
    ax.set_xticks(N_q)
    xlim0 = 45
    xlim1 = 55
    ylim0 = 55
    ylim1 = 69
    ax.plot([xlim0, xlim1, xlim1, xlim0, xlim0], [ylim0, ylim0, ylim1, ylim1, ylim0], '--', color='black', linewidth=1)
    axins.set_xlim(xlim0, xlim1)
    axins.set_ylim(ylim0, ylim1)
    xy = (xlim0, ylim0)
    xy2 = (xlim0, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)

    xy = (xlim1, ylim0)
    xy2 = (xlim1, ylim1)
    con = ConnectionPatch(xyA=xy2, xyB=xy, coordsA="data", coordsB="data",
                          axesA=axins, axesB=ax)
    axins.add_artist(con)
    Plt_set(ax, 'Number of qubits', 'NLL', 'nll_05', loc=2)

    # time
    # p=1
    fig, ax = plt.subplots(1, 1)
    for idx, model in enumerate(model_names):
        r_files = b_path + model + '/results.txt'

        time_mean = []
        time_std = []
        with open(r_files, 'r') as f:
            data = f.readlines()[:16]
            for i in range(len(data)):
                if 'Time' in data[i]:
                    time_mean.append(float(data[i][12:20]))
                    time_std.append(float(data[i][26:34]))
            print(time_mean, time_std)

        time_mean = np.array(time_mean)
        time_std = np.array(time_std)
        ax.plot(N_q, time_mean, style[idx], label=model, color=colors[idx])
        ax.fill_between(N_q, time_mean - time_std, time_mean + time_std, alpha=0.2, color=colors[idx])
    ax.set_xticks(N_q)
    Plt_set(ax, 'Number of qubits', 'Time (s)', 'time_1', loc=2)

    # p=0.5
    fig, ax = plt.subplots(1, 1)
    for idx, model in enumerate(model_names):
        r_files = b_path + model + '/results.txt'

        time_mean = []
        time_std = []
        with open(r_files, 'r') as f:
            data = f.readlines()[19:]
            for i in range(len(data)):
                if 'Time' in data[i]:
                    time_mean.append(float(data[i][12:20]))
                    time_std.append(float(data[i][26:34]))
            print(time_mean, time_std)

        time_mean = np.array(time_mean)
        time_std = np.array(time_std)
        ax.plot(N_q, time_mean, style[idx], label=model, color=colors[idx])
        ax.fill_between(N_q, time_mean - time_std, time_mean + time_std, alpha=0.2, color=colors[idx])
    ax.set_xticks(N_q)
    Plt_set(ax, 'Number of qubits', 'Time (s)', 'time_05', loc=2)