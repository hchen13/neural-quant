import json
from datetime import datetime
from pathlib import Path
from random import shuffle

import numpy as np
import pandas as pd
import requests
import tensorflow as tf
from matplotlib import pyplot as plt
from tqdm import tqdm

from dataset_management.tfannotation import read_tfrecord, create_tfrecord
from utilities.funcs import create_file_name, indicator_ATR
from utilities.visualizations import draw_trading_data


def normalize_ohlcv(data, history_size):
    anchor_price = data['close'].iloc[0]
    max_vol = data['volume'].iloc[:history_size].max()
    min_vol = data['volume'].iloc[:history_size].min()
    data[['open', 'high', 'low', 'close']] /= anchor_price
    data['volume'] = (data['volume'] - min_vol) / (max_vol - min_vol)
    if data.isnull().values.any():
        raise ValueError("[ERROR] nan detected during data generation")
    return data


def segment_trading_data(data: pd.DataFrame, n: int=144, t: int=6, norm=False, shuffle=True):
    """ Slide the given trading data and generate segments
        data:   the trading data formatted as QUANTAXIS DataStruct
        n:      history window size, default to 144 (12 hours of 5min data)
        t:      future window size, default to 6 (half hour of 5min data)
        norm:   specify whether to normalize the input data
        :return : generator of (history, future) pair
    """
    window_size = n + t
    indices = np.arange(0, len(data) - window_size + 1)
    if shuffle:
        np.random.shuffle(indices)

    for i in indices:
        segment = data.iloc[i : i + window_size].copy()
        if norm:
            # anchor_price = segment['close'].iloc[0]
            # volume_col = segment['volume']
            # anchor_volume = volume_col[volume_col > .001].min()
            # # anchor_volume = segment['volume'].iloc[0]
            # segment[['open', 'close', 'high', 'low']] /= anchor_price
            # segment['volume'] /= anchor_volume
            segment = normalize_ohlcv(segment, n)

        history = segment[:n]
        future = segment[n:]
        yield history, future


def compute_label(history: pd.DataFrame, future: pd.DataFrame, k_up=1., k_lo=None, delta_type='volatility'):
    if k_lo is None:
        k_lo = k_up
    last_bar = history.iloc[-1]
    last_price = last_bar['close']
    volatility = history['close'].std(ddof=0)
    df = indicator_ATR(history, n=len(history) // 2)
    ATR = df.iloc[-1].ATR
    delta = volatility if delta_type == 'volatility' else ATR
    upper = k_up * delta + last_price
    lower = -k_lo * delta + last_price
    above, = np.nonzero((future['close'] > upper).values)
    below, = np.nonzero((future['close'] < lower).values)
    first_above = above[0] if len(above) else len(future) + 1
    first_below = below[0] if len(below) else len(future) + 1
    if first_above < first_below:
        return 1
    elif first_below < first_above:
        return -1
    return 0


def construct_training_data(data: pd.DataFrame, n: int=144, t: int=6, k_up=1., k_lo=None, delta_type='volatility', norm=False, columns=None, shuffle=True):
    if columns is None:
        columns = ['open', 'high', 'low', 'close', 'volume']
    columns = list(columns)
    segments = segment_trading_data(data=data, n=n, t=t, norm=norm, shuffle=shuffle)
    for history, future in segments:
        y = compute_label(history, future, k_up=k_up, k_lo=k_lo, delta_type=delta_type)
        hist = history[columns].values
        fut = future[columns].values
        yield hist.astype('float32'), y, fut.astype('float32')


def save_dataset_as_tfrecords(dataset, dataset_dir, split_ratio=.1, dataset_length=None, verbose=True, buffer_size=1):
    """ Save the dataset generator into tfrecord files

    :param dataset: generator
    :param dataset_dir: root directory of the dataset to be saved
    :param split_ratio: the ratio of validation size to the total size
    :param dataset_length: optional, specified only to display the progress
    :param verbose: whether to display progress bar
    :param buffer_size: the number of samples to be saved into a single tfrecord file
    """
    train_dir = Path(str(dataset_dir)) / 'train'
    valid_dir = Path(str(dataset_dir)) / 'valid'
    train_dir.mkdir(parents=True, exist_ok=True)
    valid_dir.mkdir(parents=True, exist_ok=True)

    train_count = 0
    valid_count = 0
    train_writer = tf.io.TFRecordWriter(str(train_dir / f"{create_file_name()}.tfrecords"))
    valid_writer = tf.io.TFRecordWriter(str(valid_dir / f"{create_file_name()}.tfrecords"))
    for i, (history, y, future) in tqdm(enumerate(dataset), total=dataset_length, disable=not verbose):
        if train_count == buffer_size:
            train_writer.close()
            file_name = f"{create_file_name()}.tfrecords"
            file_path = train_dir / file_name
            train_writer = tf.io.TFRecordWriter(str(file_path))
            train_count = 0
        if valid_count == buffer_size:
            valid_writer.close()
            file_name = f"{create_file_name()}.tfrecords"
            file_path = valid_dir / file_name
            valid_writer = tf.io.TFRecordWriter(str(file_path))
            valid_count = 0
        serialized = create_tfrecord(history=history, label=y, future=future)
        if i % (1 / split_ratio) < 1:
            valid_writer.write(serialized)
            valid_count += 1
        else:
            train_writer.write(serialized)
            train_count += 1

    train_writer.close()
    valid_writer.close()


def load_dataset(directory):
    directory = Path(str(directory))
    data_files = [str(p) for p in directory.rglob('*.tfrecords')]
    print(f"[INFO] {len(data_files)} files found.", flush=True)
    shuffle(data_files)

    raw = tf.data.TFRecordDataset(data_files)
    dataset = raw.map(read_tfrecord).shuffle(buffer_size=100)
    return dataset


def visualize_label_distribution(dataset):
    labels = [-1, 0, 1]
    label_counts = {k: 0 for k in labels}

    for history, label, future in tqdm(dataset):
        label = label.numpy()
        label_counts[label] += 1

    for lb in labels:
        print(f"label {lb}: {label_counts[lb]:,d}")
    pd.DataFrame(label_counts, columns=label_counts.keys(), index=[0]).plot.bar()
    plt.show()


def display_batch(batch_data):
    history, label, future = batch_data
    history = history.numpy()[0]
    label = label.numpy()[0]
    future = future.numpy()[0]
    display_training_data(history, label, future)


def display_training_data(history, label, future, k_up=1., k_lo=None):
    N, T = len(history), len(future)
    history_df = pd.DataFrame(history, columns=['open', 'high', 'low', 'close', 'volume'])
    future_df = pd.DataFrame(future, columns=['open', 'high', 'low', 'close', 'volume'])
    df = pd.concat([history_df, future_df])
    df.index = list(map(datetime.fromtimestamp, range(N + T)))
    df.index.name = 'datetime'
    fig, axes = draw_trading_data(df, title=f"Restored data visualization, label = {label}",
                                  figsize=(12, 6))

    # 画TBM线
    if k_lo is None:
        k_lo = k_up
    volatility = history_df['close'].std(ddof=0)
    top = history_df['close'].iloc[-1] + volatility * k_up
    bot = history_df['close'].iloc[-1] - volatility * k_lo

    axes[0].axvline(x=N - .5, alpha=.4, c='white', ls='--')
    axes[0].axhline(y=top, c='g', ls='--', linewidth=1.5)
    axes[0].axhline(y=bot, c='r', ls='--', linewidth=1.5)

    plt.show()


def fetch_market_caps(sort=True):
    api_url = "https://www.binance.com/exchange-api/v2/public/asset-service/product/get-products"
    resp = requests.get(api_url, proxies={'https': '127.0.0.1:1087'})
    content = json.loads(resp.content)
    data = content['data']
    pairs = []
    for d in data:
        if d['q'] != 'USDT':
            continue
        if d['st'] != 'TRADING' or not d['cs'] or not d['c']:
            continue
        symbol = d['s']
        circulating_supply = float(d['cs'])
        price = float(d['c'])
        pairs.append({
            "symbol": symbol,
            "market_cap": circulating_supply * price,
            "circulating_supply": circulating_supply,
            "price": price,
            "vol24h": float(d['qv']),
        })
    if sort:
        pairs.sort(key=lambda i: i['market_cap'], reverse=True)
    return pairs