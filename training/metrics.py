import numpy as np
import tensorflow as tf


class MultiClassEvaluator:
    def __init__(self, labels):
        self.y_true_list = []
        self.y_pred_list = []
        self.losses = []
        self.K = len(labels)
        self.labels = labels
        self.TP = np.zeros(shape=(self.K,))
        self.FP = np.zeros(shape=(self.K,))
        self.FN = np.zeros(shape=(self.K,))

    def _convert_tensor_to_array(self, tensor):
        if isinstance(tensor, tf.Tensor):
            return tensor.numpy()
        return tensor

    def update_predictions(self, y_true, y_pred):
        y_true = self._convert_tensor_to_array(y_true)
        y_pred = self._convert_tensor_to_array(y_pred)
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)

        y_pred = tf.one_hot(np.argmax(y_pred, axis=1), depth=3).numpy()

        for k in range(self.K):
            tp = np.sum(y_true[:, k] * y_pred[:, k])
            fp = np.sum(y_pred[:, k]) - tp
            fn = np.sum(y_true[:, k]) - tp
            self.TP[k] += tp
            self.FP[k] += fp
            self.FN[k] += fn

    def update_loss(self, value):
        value = self._convert_tensor_to_array(value)
        self.losses.append(value)

    @property
    def precision(self):
        report = {}
        sum_p = 0
        for k in range(self.K):
            precision = self.TP[k] / (self.TP[k] + self.FP[k] + 1e-7)
            sum_p += precision
            report[self.labels[k]] = precision
        report['macro'] = sum_p / self.K
        # report['micro'] = np.sum(self.TP) / (np.sum(self.TP) + np.sum(self.FP) + 1e-7)
        return report

    @property
    def recall(self):
        report = {}
        sum_r = 0
        for k in range(self.K):
            recall = self.TP[k] / (self.TP[k] + self.FN[k] + 1e-7)
            sum_r += recall
            report[self.labels[k]] = recall
        report['macro'] = sum_r / self.K
        # report['micro'] = np.sum(self.TP) / (np.sum(self.TP) + np.sum(self.FN) + 1e-7)
        return report

    @property
    def f_score(self):
        report = {}
        p = self.precision
        r = self.recall
        s = 0
        for k in range(self.K):
            key_name = self.labels[k]
            f_score = 2 * (p[key_name] * r[key_name]) / (p[key_name] + r[key_name] + 1e-7)
            s += f_score
            report[key_name] = f_score
        report['macro'] = s / self.K
        # report['micro'] = 2 * (p['micro'] * r['micro']) / (p['micro'] + r['micro'] + 1e-7)
        return report

    @property
    def report(self):
        p = {f"precision/{key}": val for key, val in self.precision.items()}
        r = {f"recall/{key}": val for key, val in self.recall.items()}
        f = {f"f1-score/{key}": val for key, val in self.f_score.items()}
        report = {"loss": np.mean(self.losses), **p, **r, **f}
        report['accuracy'] = np.sum(self.TP) / (np.sum(self.TP) + np.sum(self.FN) + 1e-7)
        return report
