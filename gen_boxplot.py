# -*- coding: utf-8 -*-
"""
@file      :  gen_boxplot.py
@Time      :  2022/8/26 14:43
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""

import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt


def get_gt_files(SERVER=True):
    pt_list = []
    for file_name in os.listdir('./'):
        path = os.path.join('.', file_name)
        file = glob.glob(path + '/*.pt')
        if file:
            pt_list.extend(file)
        pt_list.sort()

    print('****** LOAD FILES: ********')
    print(pt_list)

    if not SERVER:
        pts = [torch.load(x, map_location=torch.device('cpu')) for x in pt_list]
    else:
        pts = [torch.load(x) for x in pt_list]

    print('successfully load all the pts!!!')

    return pts, pt_list


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
    # use plot function to draw a small line to name the legend.
    plt.plot([], c=color_code, label=label)
    plt.legend(fontsize=20)


def create_room_dict(temp_pt):
    room_dict = {}
    room_dict.clear()
    for key, value in temp_pt.items():
        if '.pt' not in key:
            continue
        room_config, room_name = key.split('/')[-2:]
        temp_room = room_name.split(room_config)[1].strip('_')  # 'mine_site1_1way_bformat_2'

        if temp_room not in room_dict.keys():
            # then create a new room_dict
            room_dict[temp_room] = {'gt': value['gt'].tolist(), 'mean_output': value['mean_output'].tolist()}
        else:
            room_dict[temp_room]['gt'].extend(value['gt'].tolist())
            room_dict[temp_room]['mean_output'].extend(value['mean_output'].tolist())
    room_dict = sorted(room_dict.items(), key=lambda x: x[1]['gt'][0])
    room_dict = {i[0]: i[1] for i in room_dict}
    return room_dict


def get_boxplot_clean_version(pts, names):

    room_dicts = [create_room_dict(x) for x in pts]

    epoch_nums = [m.split('/')[-1].strip('.pt') for m in names]
    color = ['#D7191C', '#2C7BB6', 'orange', 'green', 'tomato', 'forestgreen', 'royalblue', 'grey', 'lightgrey']

    """"PLOT"""

    fig = plt.figure(figsize=(40, 20))
    ticks = list(room_dicts[0].keys())

    # calculate the distribution of the boxes
    test_epochs = len(room_dicts)
    distr = [-test_epochs * 0.5 / 2 + 0.5 * i for i in range(test_epochs + 1)]

    gt_plot = plt.boxplot([v['gt'] for _, v in room_dicts[0].items()],
                          positions=np.array(np.arange(len(room_dicts[0].values()))) * test_epochs + distr[0],
                          widths=0.3)

    define_box_properties(gt_plot, 'black', 'gt')

    for i in range(test_epochs):
        temp_plot = plt.boxplot([v['mean_output'] for _, v in room_dicts[i].items()],
                                positions=np.array(np.arange(len(room_dicts[i].values()))) * test_epochs + distr[
                                    i + 1], widths=0.3)

        define_box_properties(temp_plot, color[i], 'epoch ' + epoch_nums[i])

    plt.xticks(np.arange(0, len(ticks) * test_epochs, test_epochs), ticks, fontsize=15)
    plt.yticks(fontsize=20)

    fig_name = 'Test Results'
    plt.title(fig_name, fontsize=30)  # 标题，并设定字号大小
    plt.savefig('./all_box_plot.png', dpi=fig.dpi, pad_inches=1)


if __name__ == '__main__':
    pts, pt_list = get_gt_files(SERVER=True)
    get_boxplot_clean_version(pts, pt_list)
    print('Save fig already!')

