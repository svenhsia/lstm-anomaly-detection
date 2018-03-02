import numpy as np
import pandas as pd
import tensorflow as tf

data = pd.read_csv('ft.csv')
data = data.values

##########
# Pipeline:
# 1- division dataset: train normal (T),
#                      validation normal_1 (V_n1),
#                      validation normal_2 (V_n2),
#                      test_normal (T_n)，
#                      validation abnormal （V_a),
#                      test abnormal (T_a)
# 2- scaler training set, save scaler
# 3- dimension reduction training set, save transformation matrix
# 4- train LSTM
# 5- use V_n1 to model Gaussian distribution
# 6- use V_n2 and V_a to determine liklihood threshold
# 7- test

###### A posterior, use a neural network to "explain" anomaly cause


sess = tf.Session()

s, u, v = tf.svd(data)
s = s.eval(session=sess)
v = v.eval(session=sess)

s = s ** 2
s_cum = s.cumsum()
s_cum /= s_cum[-1]  # variance percentage
dim_r = 0
while s_cum[dim_r] < 0.999:
    dim_r += 1
dim_r += 1  # number of columns to keep, dimension of new space
print(dim_r)

transform_matrix = v[:, :dim_r]
X = data.dot(transform_matrix)
assert transform_matrix.shape == (data.shape[1], dim_r)
assert X.shape == (data.shape[0], dim_r)

np.savetxt('rows.txt', X)
np.savetxt('trans_matrix.txt', transform_matrix)
