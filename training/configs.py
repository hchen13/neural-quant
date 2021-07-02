import os
from pathlib import Path

_gpu_machine = True if os.getenv('AIBOX') else False

if _gpu_machine:
    EPOCHS = 100
    BATCH_SIZE = 64
    DATASET_ROOT = Path('/media/ethan/DataStorage/quant/') / 'crypto60min'
else:
    EPOCHS = 10
    BATCH_SIZE = 16
    DATASET_ROOT = Path().home() / 'datasets' / 'quant' / 'crypto60min'

TRAIN_DIR = DATASET_ROOT / 'train'
VALID_DIR = DATASET_ROOT / 'valid'

LEARNING_RATE = 1e-5
LOG_STEPS = 20
EXAM_STEPS = 1000
CLASS_WEIGHTS = [1., .5, 1.]
