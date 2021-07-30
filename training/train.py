import sys
from datetime import datetime
from pathlib import Path

import tensorflow as tf
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE
from tqdm import tqdm

_project_root = Path().cwd().parent
sys.path.append(str(_project_root))

from dataset_management.tools import load_dataset
from training.tensorboard_monitor import Monitor
from prototype.fcn import build_fcn
from training.loss_functions import weighted_categorical_crossentropy
from training.metrics import MultiClassEvaluator
import configs
import settings


@tf.function
def train_on_batch(model, batch_data, optimizer):
    history, label, _ = batch_data
    y_true = tf.one_hot(label + 1, depth=3)
    with tf.GradientTape() as tape:
        y_pred = model(history)
        loss = weighted_categorical_crossentropy(y_true, y_pred, weights=configs.CLASS_WEIGHTS)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    report = {'loss': loss}
    return report


def examine(model, dataset, limit=None):
    evaluator = MultiClassEvaluator(labels=['-1', '0', '1'])
    for i, (history, label, future) in enumerate(dataset):
        y_true = tf.one_hot(label + 1, depth=3)
        y_pred = model.predict(history)
        loss = weighted_categorical_crossentropy(y_true, y_pred, weights=configs.CLASS_WEIGHTS)
        evaluator.update_loss(loss)
        evaluator.update_predictions(y_true, y_pred)

        if limit is not None and i >= limit:
            break

    return evaluator.report


if __name__ == '__main__':
    print(f"[INFO] Loading dataset...")
    trainset = load_dataset(configs.TRAIN_DIR)
    validset = load_dataset(configs.VALID_DIR)
    trainset = trainset.batch(configs.BATCH_SIZE).prefetch(AUTOTUNE)
    validset = validset.batch(configs.BATCH_SIZE).prefetch(AUTOTUNE)

    print(f"[INFO] Creating model for training...")
    fcn = build_fcn(input_size=None)
    # fcn.load_weights(str(settings.WEIGHTS_DIR / 'fcn_crypto.h5'))

    print(f"[INFO] Setting up monitoring and learning algorithm...")
    experiment_name = f"FCN-crypto1h@{datetime.now().strftime('%-y%m%d-%H:%M:%S')}"
    monitor = Monitor(experiment_name)
    optimizer = tf.keras.optimizers.Adam(learning_rate=configs.LEARNING_RATE)

    print(f"[INFO] Start training...")
    global_step = 0
    for e in range(configs.EPOCHS):
        print(f"epoch {e + 1}/{configs.EPOCHS} @{datetime.now().strftime('%y-%m-%d %H:%M:%S')}")
        for local_step, batch_data in tqdm(enumerate(trainset)):
            global_step += 1
            train_report = train_on_batch(fcn, batch_data, optimizer)

            if global_step % configs.LOG_STEPS == 0:
                monitor.write_reports(train_report, global_step, prefix='train_')
                monitor.write_weights(fcn, step=global_step)

            if global_step % configs.EXAM_STEPS == 0:
                exam_report = examine(fcn, validset, limit=global_step // 100)
                monitor.write_reports(exam_report, global_step, prefix='valid_')
            '''------ end of epoch ------'''

        print(f"[INFO] Saving intermediate weights at epoch #{e + 1}")
        tmp_path = settings.OUTPUT_DIR / 'tmp'
        tmp_path.mkdir(exist_ok=True, parents=True)
        fcn.save_weights(str(tmp_path / f"tmp.h5"))
        ''' ----- end of training ------'''

    print(f"[INFO] Training complete, saving final weights: {configs.OUTPUT_WEIGHT_FILE}")
    save_path = settings.WEIGHTS_DIR / configs.OUTPUT_WEIGHT_FILE
    fcn.save_weights(str(save_path))
    print(f"[INFO] BYE!")
