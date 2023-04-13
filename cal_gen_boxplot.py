# -*- coding: utf-8 -*-
"""
@file      :  new_gen_curve.py
@Time      :  2022/10/28 17:58
@Software  :  PyCharm
@summary   :  Using all the data to plot
@Author    :  Bajian Xiang
"""

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import MultipleLocator
import glob

csv_file = glob.glob('./*.csv')[0]

x = pd.read_csv(csv_file)


def get_analysis(cc):
    print('get_analysis...')
    data = {}
    print('len(cc)', len(cc))
    for i in range(len(cc)):
        if "/data/" in cc.iloc[i, 0]:
            room = cc.iloc[i, 0].split('/')[-1].strip('.pt')[:-2]

            m = i + 1
            while cc.iloc[m, 0] != 'mean_output':
                m += 1
            if room not in data.keys():
                data[room] = {}
                data[room]['mean'] = float(cc.iloc[m, 1])
                data[room]['gt'] = float(cc.iloc[m + 1, 1])

                n = i + 1
                data[room]['value'] = []
                while cc.iloc[n, 0] != 'mean_output':
                    if 'output' in cc.iloc[n, 0] and 'mean' not in cc.iloc[n, 0]:
                        data[room]['value'].append(float(cc.iloc[n, 1]))
                    n += 1
            else:
                temp_mean = float(cc.iloc[m, 1])
                data[room]['mean'] = (data[room]['mean'] + temp_mean) / 2

                if data[room]['gt'] != float(cc.iloc[m + 1, 1]):
                    print('origin:', cc.iloc[i, 0])
                    print('wrong in room:', room)

                n = i + 1
                while cc.iloc[n, 0] != 'mean_output':
                    if 'output' in cc.iloc[n, 0] and 'mean' not in cc.iloc[n, 0]:
                        data[room]['value'].append(float(cc.iloc[n, 1]))
                    n += 1
        if i % 10000 == 0 : print(i)

    csv = pd.DataFrame.from_dict(data).T

    return csv.sort_values(by='gt')

def lst_mean(lst):
    total = 0
    for value in lst:
        total += value
    return total/len(lst)

def plot(csv):
    print('begin plot')
    plt.rc('font', family='Times New Roman')
    fig = plt.figure(figsize=(30, 22))
    # mean = csv.loc[:, 'mean'].tolist()
    # gt = csv.loc[:, 'gt'].tolist()
    # value = csv.loc[:, 'value'].tolist()
    # room = csv.index.tolist()
    i = 0
    mean = [] # 0
    gt = []   # 1
    value = [] # 2
    print('lencsv:',len(csv))
    while i <= len(csv)-1:
        temp_gt = csv.iloc[i, 1] * 100 // 1 / 100 # 0.51xxxx -> 51 -> 0.51
        #if temp_gt < 0.25 or temp_gt > 0.85:
        #    i += 1
        #    continue
        if not gt:
            gt.append(temp_gt)
            temp_mean = [csv.iloc[i, 0]]
            value.append(csv.iloc[i, 2])
        if temp_gt != gt[-1]:
            # temp_mean 给上一个
            mean.append(lst_mean(temp_mean))
            temp_mean = [csv.iloc[i, 0]] # 清零
            gt.append(temp_gt)
            value.append(csv.iloc[i, 2])
        else:
            temp_mean.append(csv.iloc[i, 0])
            value[-1].extend(csv.iloc[i, 2])
        i = i + 1
        if i % 10000 == 0 : print(i, end=',')
    mean.append(lst_mean(temp_mean))

    plt.boxplot(value, showfliers=False, positions=[i*100 for i in gt])# Outlier
    print('len of mean and gt:', len(mean), len(gt))
    print('len of value:', len(value))
    ax = plt.gca()
    print('gt:', gt)
    print('mean:', mean)
    assert len(gt) <= 100
    memory = {}
    for i, tempp in enumerate(value):
        le = len(tempp)
        tempp.sort()
        memory[gt[i]] = {'mean':mean[i], 'q1':tempp[int(le/4)], 'median':tempp[int(le/2)],'q3':tempp[int(le/4*3)]}
    print(len(memory))
    pd.DataFrame.from_dict(memory).T.to_csv('calculated_data.csv')
    y_major_locator = MultipleLocator(0.1)
    x_major_locator = MultipleLocator(10)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.xaxis.set_major_locator(x_major_locator)
    #gt.insert(0, 0.29)
    #mean.insert(0, 0.29)
    x_locate = [ i * 100 for i in gt]
    p1, = plt.plot(x_locate, gt, color='red', marker='.', markersize=15, linewidth=3)
    p2, = plt.plot(x_locate, mean, color='royalblue', marker='.', markersize=15, linewidth=3)
    plt.ylabel('STI prediction', fontsize=40)
    plt.xlabel('STI ground truth', fontsize=40)

    #labels = [item.get_text() for item in ax.get_xticklabels()]
    #print('original lable', labels)
    #for i in range(len(labels)):
    #    labels[i] = str(0.05 * (i+4))[:4]
    labels = ['0.2',  '0.3', '0.4', '0.5', '0.6', '0.7', '0.8', '0.9']
    ax.set_xticklabels(labels)
    #plt.ylim(0.2, 0.96)
    plt.tick_params(labelsize=30)
    plt.title('Estimating the speech transmission index', fontsize=45)
    plt.grid(linestyle='--', linewidth=0.5)
    plt.xticks(rotation=0, fontsize=30)
    plt.legend([p1, p2], ['Ideal estimation', 'Proposed method'], fontsize=38)
    # plt.show()
    fig.savefig('./0409_all_Freq_without_outlier.png')
    print('savefig')

print('csv_file:', csv_file)
csv = get_analysis(x)
plot(csv)

