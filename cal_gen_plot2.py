# -*- coding: utf-8 -*-
"""
@file      :  gen_full_freq_boxplot.py
@Time      :  2022/12/8 20:41
@Software  :  PyCharm
@summary   :
@Author    :  Bajian Xiang
"""
import os

import pandas as pd
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator

def get_csvs():
    csv_lst = []
    for file_name in os.listdir('../../'):
        path = os.path.join('../..', file_name)
        csv_file = glob.glob(path + '/*.csv')
        if csv_file:
            csv_lst.extend(csv_file)

    print('******** LOAD FILES ********')
    print(csv_lst)
    csvs = [get_analysis(pd.read_csv(i), i) for i in csv_lst]
    print('successfully load all the csvs!!!')
    return csvs, csv_lst


def get_analysis(cc, name):
    print('current csv:', name)
    data = {}
    for i in range(len(cc)):
        if "/data/" in cc.iloc[i, 0]:
            config = cc.iloc[i, 0].split('/')[-2]
            room = cc.iloc[i, 0].split(config)[-2].strip('_')
            m = i + 1

            while cc.iloc[m, 0] != 'mean_output':
                m += 1
            if room not in data.keys():
                data[room] = {}
                data[room]['gt'] = {'500': [float(cc.iloc[m + 1, 1])], '1k': [float(cc.iloc[m + 1, 2])],
                                    '2k': [float(cc.iloc[m + 1, 3])], '4k': [float(cc.iloc[m + 1, 4])]}

                n = i + 1
                data[room]['value'] = {'500': [], '1k': [], '2k': [], '4k': []}
                while cc.iloc[n, 0] != 'mean_output':
                    if 'output' in cc.iloc[n, 0] and 'mean' not in cc.iloc[n, 0]:
                        data[room]['value']['500'].append(float(cc.iloc[n, 1]))
                        data[room]['value']['1k'].append(float(cc.iloc[n, 2]))
                        data[room]['value']['2k'].append(float(cc.iloc[n, 3]))
                        data[room]['value']['4k'].append(float(cc.iloc[n, 4]))
                    n += 1
            else:
                n = i + 1
                while cc.iloc[n, 0] != 'mean_output':
                    if 'output' in cc.iloc[n, 0] and 'mean' not in cc.iloc[n, 0]:
                        data[room]['value']['500'].append(float(cc.iloc[n, 1]))
                        data[room]['value']['1k'].append(float(cc.iloc[n, 2]))
                        data[room]['value']['2k'].append(float(cc.iloc[n, 3]))
                        data[room]['value']['4k'].append(float(cc.iloc[n, 4]))
                    n += 1

    for freq in ['500', '1k', '2k', '4k']:
        count = 0
        total_output = 0
        total_gt = 0
        total_se = 0
        for room in data.keys():
            temp_gt = data[room]['gt'][freq][0]
            temp_count = len(data[room]['value'][freq])
            temp_room_output = sum(data[room]['value'][freq])
            temp_room_bias = temp_room_output - temp_count * temp_gt
            temp_room_se = sum([(temp-temp_gt)**2 for temp in data[room]['value'][freq]])
            print('** room: %s\t count:%d \t mean bias:%.4f \t mean mse:%.4f' % (room[:7], temp_count, temp_room_bias/temp_count, temp_room_se/temp_count))

            total_output += temp_room_output
            count += temp_count
            total_gt += temp_gt * temp_count
            total_se += temp_room_se

        mean_bias = (total_output - total_gt) / count
        mean_mse = total_se / count
        print('freq', freq, '\tmean_bias:', mean_bias, '\tmean_mse:', mean_mse)

    data = sorted(data.items(), key=lambda x: x[1]['gt'])
    data = {i[0]: i[1] for i in data}
    return data


def define_box_properties(plot_name, color_code, label):
    for k, v in plot_name.items():
        plt.setp(plot_name.get(k), color=color_code)
    plt.plot([], c=color_code, label=label)
    plt.legend(fontsize=20)


def get_boxplot_clean_version(room_dicts, names, freq):
    color = ['#D7191C', '#2C7BB6', 'orange', 'green', 'tomato', 'forestgreen', 'royalblue', 'grey', 'lightgrey']

    fig = plt.figure(figsize=(40, 20))
    ticks = list(room_dicts[0].keys())
    test_epochs = len(room_dicts)
    distr = [-test_epochs * 0.5 / 2 + 0.5 * i for i in range(test_epochs + 1)]
    epoch_nums = [m.split('epoch')[-1].strip('.csv') for m in names]

    """"PLOT"""
    gt_plot = plt.boxplot([v['gt'][freq] for _, v in room_dicts[0].items()],
                          positions=np.array(np.arange(len(room_dicts[0].values()))) * test_epochs + distr[0],
                          widths=0.3)
    define_box_properties(gt_plot, 'black', 'gt')

    for i in range(test_epochs):
        temp_plot = plt.boxplot([v['value'][freq] for _, v in room_dicts[i].items()],
                                positions=np.array(np.arange(len(room_dicts[i].values()))) * test_epochs + distr[
                                    i + 1], widths=0.3)

        define_box_properties(temp_plot, color[i], 'epoch ' + epoch_nums[i])

    plt.xticks(np.arange(0, len(ticks) * test_epochs, test_epochs), ticks, fontsize=15, rotation=30)
    plt.yticks(fontsize=20)
    plt.ylim(0.0, 1.5)
    plt.grid(linestyle='-.')
    y_major_locator = MultipleLocator(0.1)
    ax = plt.gca()
    ax.yaxis.set_major_locator(y_major_locator)
    fig_name = 'Test Results'
    plt.title(fig_name, fontsize=30)  # 标题，并设定字号大小
    plt.savefig('./new_all_boxplot_' + freq + 'hz.png', dpi=fig.dpi, pad_inches=1)
    print('freq saved:', freq)


if __name__ == "__main__":
    csvs, csv_names = get_csvs()
    for freq in ['500', '1k', '2k', '4k']:
        get_boxplot_clean_version(csvs, csv_names, freq)

