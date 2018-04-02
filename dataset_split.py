import pandas as pd
import numpy as np
from numpy.random import permutation


def periodInfo(path):
    """
        Periods' information after the missing vector prediction.
    """
    ft = pd.read_csv(path + 'final_feature_table.csv')
    ft.set_index('timestamp', inplace=True)
    periods = []
    broken = True
    print ('\nperiods info')
    for i in ft.index:
        if broken:
            start = i
            broken = False
        else:
            end = i
        if i+1 not in ft.index:
            periods.append((start, end))
            broken = True
            print("continuous between ({}, {}) for {} timesteps".format(start, end, end-start+1))
    return ft, periods


def extractAnomalies(period, periods_a):
    """
        Given periods and abnormal periods, calculate normal periods.
    """
    contains_a = False
    for pa in periods_a:
        if period[0]<=pa[0] and period[1]>=pa[1]:
            contains_a = True
            return extractAnomalies((period[0], pa[0]-1), periods_a) +\
                   extractAnomalies((pa[1]+1, period[1]), periods_a)
    if not contains_a:
        return [period]


def flattenList(l):
    """
        Flatten a list of lists to a list.
    """
    return [item for subl in l for item in subl]


def extractPeriod(path):
    """
        Load periods and abnormal periods.
        Calculate normal periods.
    """
    # load abnormal periods
    abnormal = pd.read_csv(path + 'abnormal_treated.csv')
    periods_a = [(abnormal.iloc[i, 1], abnormal.iloc[i, 2]) for i in range(abnormal.shape[0])]
    # load periods
    ft, periods = periodInfo(path)
    periods_n = []
    # extract normal periods
    for p in periods:
        periods_n += extractAnomalies(p, periods_a)
    print ('\nabnormal periods ', periods_a)
    print ('\nnormal periods')
    for pn in periods_n:
        print("from {} to {} covering {} steps".format(pn[0], pn[1], pn[1]-pn[0]+1))
    return ft, periods_n, periods_a


def splitIndice(periods_n, periods_a, path):
    """
        Calculate the indices for splitting datasets.
        Calculate the splitted periods.
    """
    # calculate the indices of 4 datasets for normal periods
    periods_split = [[], [], [], []]
    proportions = {0:0.6, 1:0.15, 2:0.15, 3:0.1}
    for p in periods_n:
        len_p = p[1] - p[0] + 1
        perm = permutation(4)
        cut_pos = [p[0]-1] + [int(len_p*proportions[idx]) for idx in perm]
        cut_pos = np.array(cut_pos).cumsum()
        cut_pos[-1] = p[1]   
        for idx, no_split in enumerate(perm):
            periods_split[no_split].append((cut_pos[idx]+1, cut_pos[idx+1]))
    indice_train = [[i for i in range(p[0], p[1]+1)] for p in periods_split[0]]
    indice_val1 = [[i for i in range(p[0], p[1]+1)] for p in periods_split[1]]
    indice_val2 = [[i for i in range(p[0], p[1]+1)] for p in periods_split[2]]
    indice_testn = [[i for i in range(p[0], p[1]+1)] for p in periods_split[3]]
    # calculate the indices of 2 datasets for abnormal periods
    periods_a_split = [((p[0], (p[0]+p[1])//2), ((p[0]+p[1])//2+1, p[1])) for p in periods_a]
    periods_a_split = [[p[0] for p in periods_a_split], [p[1] for p in periods_a_split]]
    indice_vala = [[i for i in range(p[0], p[1]+1)] for p in periods_a_split[0]]
    indice_testa = [[i for i in range(p[0], p[1]+1)] for p in periods_a_split[1]]
    # save the splitted periods info
    with open(path + 'periods_split.txt', 'w') as f:
        f.write("normal periods:\n")
        for subl in periods_split:
            f.write("{}\n".format(subl))
        f.write("abnormal periods:\n")
        for subl in periods_a_split:
            f.write("{}\n".format(subl))
    return indice_train, indice_val1, indice_val2, indice_testn, indice_vala, indice_testa


def split(ft, indice_train, indice_val1, indice_val2, indice_testn, indice_vala, indice_testa, path):
    """
        Split datasets using the indices calculated before.
    """
    # split into 4 datasets for normal periods
    train = ft.loc[flattenList(indice_train), :]
    train.to_csv(path + 'train.csv')
    val1 = ft.loc[flattenList(indice_val1), :]
    val1.to_csv(path + 'val1.csv')
    val2 = ft.loc[flattenList(indice_val2), :]
    val2.to_csv(path + 'val2.csv')
    testn = ft.loc[flattenList(indice_testn), :]
    testn.to_csv(path + 'testn.csv')
    # split into 2 datasets for abnormal periods
    vala = ft.loc[flattenList(indice_vala), :]
    vala.to_csv(path + 'vala.csv')
    testa = ft.loc[flattenList(indice_testa), :]
    testa.to_csv(path + 'testa.csv')





path = '../data/'
ft, periods_n, periods_a = extractPeriod(path)
indice_train, indice_val1, indice_val2, indice_testn, indice_vala, indice_testa = splitIndice(periods_n, periods_a, path)
split(ft, indice_train, indice_val1, indice_val2, indice_testn, indice_vala, indice_testa, path)

