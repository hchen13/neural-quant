import os
from pathlib import Path

_gpu_machine = True if os.getenv('AIBOX') else False

if _gpu_machine:
    EPOCHS = 200
    BATCH_SIZE = 128
    DATASET_ROOT = Path('/media/ethan/DataStorage/quant/')
else:
    EPOCHS = 10
    BATCH_SIZE = 16
    DATASET_ROOT = Path().home() / 'datasets' / 'quant'

DATASET_ROOT = DATASET_ROOT / 'crypto60min_std'
TRAIN_DIR = DATASET_ROOT / 'train'
VALID_DIR = DATASET_ROOT / 'valid'

LEARNING_RATE = 1e-5
LOG_STEPS = 200
EXAM_STEPS = 1000
CLASS_WEIGHTS = [1, .4, 1]

OUTPUT_WEIGHT_FILE = 'crypto1h_std'
