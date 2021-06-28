import numpy as np
import tensorflow as tf

from training.loss_functions import weighted_categorical_crossentropy


y_true = [0, 0, 0, 1, 1, 1, 2, 2, 2]
y_pred = [0, 1, 2, 0, 1, 2, 0, 1, 2]

y_true = tf.one_hot(y_true, depth=3).numpy()
y_pred = tf.one_hot(y_pred, depth=3).numpy()

print(y_true)
print(y_pred)

for k in range(3):
    TP = np.sum(y_true[:, k] * y_pred[:, k])
    FN = np.sum(y_true[:, k]) - TP
    FP = np.sum(y_pred[:, k]) - TP
    print(f"TP{k}={TP} FP{k}={FP} FN{k}={FN} TN{k}={TN}")

