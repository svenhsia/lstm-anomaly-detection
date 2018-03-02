# Using LSTM to detect anomalies of Hadoop system
This is a course project.

## Pipeline

- division dataset: train normal (T),
                    validation normal_1 (V_n1),
                    validation normal_2 (V_n2),
                    test_normal (T_n)，
                    validation abnormal （V_a),
                    test abnormal (T_a)
- scaler training set, save scaler
- dimension reduction training set, save transformation matrix
- train LSTM
- use V_n1 to model Gaussian distribution
- use V_n2 and V_a to determine liklihood threshold
- test

If there is enough time, try using a neural network to "explain" anomaly causes.