import numpy as np
import pandas as pd

def createSequnces(dataset, lookback=10):
    """Transform a continuous data sequence to 
    dataX in shape (num_instances, lookback, num_features)
    and dataY in shape (num_instances, num_features)"""
    data = dataset.values
    dataX, dataY = [], []
    for i in range(len(data)-lookback):
        x = data[i:(i+lookback)]
        dataX.append(x)
        y = data[i+lookback]
        dataY.append(y)
    return dataX, dataY

def getBrokens(indice):
    """Get the broken positions of a discontinuous sequnence"""
    broken_points = [0]
    for i, ts in enumerate(indice):
        if ts+1 not in indice:
            broken_points.append(i+1)
    return broken_points

def createDataset(dataset, lookback):
    """Divide a discountinuous sequence into several continuous sequences 
    and transform each of them to dataX and dataY form,
    then concatenate all together"""
    X, Y = [], []
    broken_points = getBrokens(dataset.index)
    for i in range(len(broken_points)-1):
        x, y = createSequnces(dataset.iloc[broken_points[i]:broken_points[i+1], :], lookback)
        X += x
        Y += y
    return np.array(X), np.array(Y)

def pointError(trueY, predY):
    trueY_ = np.array(trueY)
    predY_ = np.array(predY)
    rst = ((predY_ - trueY_) ** 2).mean(axis=1)
    return rst

def windowAverage(errors, window=5):
    errors_ = np.array(errors)
    rst = [errors_[i:(i+window)].mean() for i in range(len(errors)-window+1)]
    return rst

def getOptimalFscore(y_true, prob_pred, order='desc', beta=1):
    arg_sorted = np.argsort(prob_pred)
    arg_sorted = arg_sorted[::-1] if order=='desc' else arg_sorted
    y_true_sort = np.array(y_true)[arg_sorted]
    prob_sort = np.array(prob_pred)[arg_sorted]

    p_best, r_best, f1_best, ts = 0, 0, 0, 0
    num_ones = sum(y_true)
    count = 0
    for i, label in enumerate(y_true_sort):
        if label == 1.:
            count += 1
        p = count / (i+1)
        r = count / num_ones
        if count == 0:
            f1 = 0
        else:
            f1 = (1 + beta**2) * p * r / (beta**2 * p + r)        
        if f1 > f1_best:
            p_best = p
            r_best = r
            f1_best = f1
            ts = prob_sort[i]
    return p_best, r_best, f1_best, ts
