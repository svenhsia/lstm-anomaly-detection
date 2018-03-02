import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

train = pd.read_csv('../train.csv', index_col=0)
train = train.values

scaler = StandardScaler()
scaler.fit(train)
joblib.dump(scaler, '../scaler.save')

train = scaler.transform(train)

# svd
sess = tf.Session()

s, u, v = tf.svd(train)
s = s.eval(session=sess)
v = v.eval(session=sess)

s = s ** 2
s_cum = s.cumsum()
s_cum /= s_cum[-1]  # variance percentage
dim_r = 0
while s_cum[dim_r] < 0.98:
    dim_r += 1
dim_r += 1  # number of columns to keep, dimension of new space
print(dim_r)

transform_matrix = v[:, :dim_r]
X_train = train.dot(transform_matrix)
assert transform_matrix.shape == (train.shape[1], dim_r)
assert X_train.shape == (train.shape[0], dim_r)

np.savetxt('../X_train.txt', X_train)
np.savetxt('../trans_matrix.txt', transform_matrix)
