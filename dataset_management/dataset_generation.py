import sys
from pathlib import Path

import QUANTAXIS as qa
from tqdm import tqdm

_project_root = Path().cwd().parent
sys.path.append(str(_project_root))

from dataset_management.tools import construct_training_data, load_dataset, visualize_label_distribution, \
    save_dataset_as_tfrecords, display_training_data

if __name__ == '__main__':
    # freq = '60min'
    # start_date = "2017-01-01"
    # end_date = "2020-12-31"
    #
    # dataset_dir = Path().home() / 'datasets' / 'quant' / f'crypto{freq}'
    # split_ratio = .05
    #
    # params = dict(
    #     n=144,
    #     t=12,
    #     k_up=1.,
    #     norm=True
    # )
    #
    # size = 0
    # coins = [
    #     'dogeusdt', 'btcusdt', 'ethusdt', 'eosusdt', 'xrpusdt',
    #     'ltcusdt', 'trxusdt', 'linkusdt'
    # ]
    # for coin in coins:
    #     symbol = f'HUOBI.{coin}'
    #     print(f"[INFO] Loading {symbol}@{freq} data: {start_date} ~ {end_date}...")
    #     data = qa.QA_fetch_cryptocurrency_min_adv(symbol, start=start_date, end=end_date, frequence=freq).data.dropna()
    #     dataset = construct_training_data(data, **params)
    #     dataset_length = len(data) - (params['n'] + params['t']) + 1
    # #
    #     print(f"[INFO] creating tfrec files", flush=True)
    #     save_dataset_as_tfrecords(dataset, dataset_dir, split_ratio=split_ratio, dataset_length=dataset_length)
    """
    label -1: 26,952
    label 0: 128,513
    label 1: 29,224
    9726it [00:04, 2054.48it/s]
    label -1: 1,449
    label 0: 6,768
    label 1: 1,509
    """

    start_date = '2018-01-01'
    end_date = '2018-12-31'
    params = dict(
        n=120,
        t=10,
        k_up=1.,
        norm=True
    )
    dataset_dir = Path().home() / 'datasets' / 'quant' / f'stock1d'
    split_ratio = .001
    stock_list = qa.QA_fetch_stock_list_adv()
    for i, (_, row) in tqdm(enumerate(stock_list.iterrows()), total=len(stock_list), disable=False):
        symbol = row['code']
        queryset = qa.QA_fetch_stock_day_adv(symbol, start=start_date, end=end_date)
        if queryset is None or len(queryset) < (params['n'] + params['t']):
            continue
        # print(f"({i + 1}/{len(stock_list)}): {row['name']}({symbol:s})", end='')
        candlesticks = queryset.to_qfq().data.dropna()
        dataset_length = len(candlesticks) - (params['n'] + params['t']) + 1
        dataset = construct_training_data(candlesticks, **params, shuffle=True)
        save_dataset_as_tfrecords(
            dataset, dataset_dir,
            split_ratio=split_ratio,
            dataset_length=dataset_length,
            verbose=False, buffer_size=1024)
        break
    ''' stock day 2015-2017 label distribution
    label -1: 108,339
    label 0: 593,256
    label 1: 110,377
    '''


    trainset = load_dataset(dataset_dir / 'train')
    validset = load_dataset(dataset_dir / 'valid')

    visualize_label_distribution(trainset)
    visualize_label_distribution(validset)
