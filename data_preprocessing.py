import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
import matplotlib.pyplot as plt


def dimensionReduction(X_train, threshold):
    """
        Do svd.
        Calculate the retained variance for different dimension.
        Determine the reduced dimension.
        Transform the train set and reduce its dimension using a transformation matrix calculated.
    """
    # calculate the retained variance
    sess = tf.Session()
    s, u, v = tf.svd(X_train)
    s = s.eval(session=sess)
    v = v.eval(session=sess)
    s = s ** 2
    s_cum = s.cumsum()
    s_cum /= s_cum[-1]
    # draw the picture of change of retained variance with reduced dimension
    plt.plot(s_cum)
    plt.xlabel('reduced dimension')
    plt.ylabel('retained variance')
    plt.grid(True)
    plt.show()
    # calculate the reduced dimension
    dim_r = 0
    while s_cum[dim_r] < threshold:
        dim_r += 1
    dim_r += 1
    print ('\nreduced dimension ', dim_r)
    # calculate the transformation matrix
    trans_matrix = v[:, :dim_r]
    np.savetxt(path + 'trans_matrix.txt', trans_matrix)
    # transform train set
    X_train = X_train.dot(trans_matrix)
    return X_train, trans_matrix


def transform(df, scaler, trans_matrix, scaler2, filename):
    """
        Given fitted scalers and transformation matrix, 
        reduce the dimension of validation sets and test sets, 
        and normalize them. 
    """
    df_index = df.index
    df = scaler.transform(df)
    df = df.dot(trans_matrix)
    df = scaler2.transform(df)
    df = pd.DataFrame(df, index=df_index)
    df.to_csv(path + filename)





# load train set
path = '../data/'
train = pd.read_csv(path + 'train.csv', index_col=0)
train_index = train.index

# scaling
scaler = MinMaxScaler()
X_train = scaler.fit_transform(train)
joblib.dump(scaler, path + 'scaler.save')

# reduce dimension
X_train, trans_matrix = dimensionReduction(X_train, 0.99)
X_train = pd.DataFrame(X_train, index=train_index)

# scaling
scaler2 = MinMaxScaler()
X_train = scaler2.fit_transform(X_train)
joblib.dump(scaler2, path + 'scaler2.save')

# save result
X_train = pd.DataFrame(X_train, index=train_index)
X_train.to_csv(path + 'train_treated.csv')

# apply to other datasets
val1 = pd.read_csv(path + 'val1.csv', index_col=0)
transform(val1, scaler, trans_matrix, scaler2, 'val1_treated')
val2 = pd.read_csv(path + 'val2.csv', index_col=0)
transform(val2, scaler, trans_matrix, scaler2, 'val2_treated')
testn = pd.read_csv(path + 'testn.csv', index_col=0)
transform(testn, scaler, trans_matrix, scaler2, 'testn_treated')
vala = pd.read_csv(path + 'vala.csv', index_col=0)
transform(vala, scaler, trans_matrix, scaler2, 'vala_treated')
testa = pd.read_csv(path + 'testa.csv', index_col=0)
transform(testa, scaler, trans_matrix, scaler2, 'testa_treated')


