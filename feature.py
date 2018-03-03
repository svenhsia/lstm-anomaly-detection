import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib

train = pd.read_csv('../train.csv', index_col=0)
print(train.head())
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
np.savetxt('../trans_matrix.txt', transform_matrix)

X_train = train.dot(transform_matrix)
np.savetxt('../train.txt', X_train)

assert transform_matrix.shape == (train.shape[1], dim_r)
assert X_train.shape == (train.shape[0], dim_r)

val_n1 = pd.read_csv('../val_n1.csv', index_col=0).values
val_n1 = scaler.transform(val_n1)
val_n1 = val_n1.dot(transform_matrix)
np.savetxt('../val_n1.txt', val_n1)

val_n2 = pd.read_csv('../val_n2.csv', index_col=0).values
val_n2 = scaler.transform(val_n2)
val_n2 = val_n2.dot(transform_matrix)
np.savetxt('../val_n2.txt', val_n2)

test_n = pd.read_csv('../test_n.csv', index_col=0).values
test_n = scaler.transform(test_n)
test_n = test_n.dot(transform_matrix)
np.savetxt('../test_n.txt', test_n)

val_a = pd.read_csv('../val_a.csv', index_col=0).values
val_a = scaler.transform(val_a)
val_a = val_a.dot(transform_matrix)
np.savetxt('../val_a.txt', val_a)

test_a = pd.read_csv('../test_a.csv', index_col=0).values
test_a = scaler.transform(test_a)
test_a = test_a.dot(transform_matrix)
np.savetxt('../test_a.txt', test_a)
