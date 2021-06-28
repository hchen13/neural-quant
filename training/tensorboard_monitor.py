import shutil
import tensorflow as tf

from settings import PROJECT_ROOT


class Monitor:
    def __init__(self, caption):
        log_root = PROJECT_ROOT / 'training' / 'logs'
        fullpath = log_root / caption
        try:
            shutil.rmtree(str(fullpath))
        except FileNotFoundError:
            pass
        fullpath.mkdir(exist_ok=True, parents=True)
        self.logdir = fullpath
        self.caption = caption
        train_path = fullpath / 'train'
        valid_path = fullpath / 'valid'
        self.train_writer = tf.summary.create_file_writer(str(train_path))
        self.valid_writer = tf.summary.create_file_writer(str(valid_path))

    def scalar(self, tag, value, step):
        if tag.startswith('train_'):
            writer = self.train_writer
            tag = tag[len('train_'):]
        else:
            writer = self.valid_writer
            if tag.startswith('valid_'):
                tag = tag[len('valid_'):]
        with writer.as_default():
            tf.summary.scalar(tag, data=value, step=step)

    def write_reports(self, results, step, prefix=None):
        tags = results
        if prefix is not None:
            tags = { f"{prefix}{k}": v for k, v in results.items() }
        for key, val in tags.items():
            self.scalar(key, val, step)

    def graph(self, model):
        from tensorflow.python.ops import summary_ops_v2
        from tensorflow.python.keras import backend as K

        with self.train_writer.as_default():
            with summary_ops_v2.always_record_summaries():
                if not model.run_eagerly:
                    summary_ops_v2.graph(K.get_graph(), step=0)

    def write_weights(self, model, step):
        with self.train_writer.as_default():
            for weights in model.trainable_weights:
                tf.summary.histogram(weights.name, weights.numpy(), step=step)

    def write_gradients(self, gradients, weights, step):
        with self.train_writer.as_default():
            for g, w in zip(gradients, weights):
                name = f'{w.name}-grads'
                tf.summary.histogram(name, g.numpy(), step=step)
                name = f"gradients/{w.name}"
                average_grad = tf.reduce_mean(g)
                tf.summary.scalar(name, data=average_grad, step=step)