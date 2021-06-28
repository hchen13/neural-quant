from pathlib import Path

import QUANTAXIS as qa
from tqdm import tqdm

from dataset_management.tools import construct_training_data, load_dataset, visualize_label_distribution, \
    save_dataset_as_tfrecords
from utilities.visualizations import display_trading_data

_project_root = Path().cwd().parent


if __name__ == '__main__':
    # freq = '5min'
    # start_date = "2019-01-01"
    # end_date = "2021-01-01"
    #
    # dataset_dir = Path().home() / 'datasets' / 'quant' / 'btc_2019_2020'
    # split_ratio = .1
    #
    # params = dict(
    #     n=144,
    #     t=12,
    #     k_up=1.,
    #     norm=True
    # )

    # symbol = 'HUOBI.btcusdt'
    # print(f"[INFO] Loading {symbol}@{freq} data: {start_date} ~ {end_date}...")
    # data = qa.QA_fetch_cryptocurrency_min_adv(symbol, start=start_date, end=end_date, frequence=freq).data.dropna()
    # dataset = construct_training_data(data, **params)
    # dataset_length = len(data) - (params['n'] + params['t']) + 1
    #
    # print(f"[INFO] creating tfrec files", flush=True)
    # save_dataset_as_tfrecords(dataset, dataset_dir, split_ratio=split_ratio, dataset_length=dataset_length)
    ''' btc 5min 2019-2020 label distribution
    label -1: 34,196
    label 0: 118,880
    label 1: 36,518
    '''


    start_date = '2015-01-01'
    end_date = '2017-01-01'
    params = dict(
        n=120,
        t=12,
        k_up=1.,
        norm=True
    )
    dataset_dir = Path().home() / 'datasets' / 'quant' / f'stock_2015_2017'
    split_ratio = .001
    stock_list = qa.QA_fetch_stock_list_adv()
    for i, (_, row) in tqdm(enumerate(stock_list.iterrows()), total=len(stock_list), disable=False):
        symbol = row['code']
        queryset = qa.QA_fetch_stock_day_adv(symbol, start=start_date, end=end_date)
        if queryset is None or len(queryset) < 100:
            continue
        # print(f"({i + 1}/{len(stock_list)}): {row['name']}({symbol:s})", end='')
        candlesticks = queryset.to_qfq().data.dropna()
        dataset_length = len(candlesticks) - (params['n'] + params['t']) + 1
        dataset = construct_training_data(candlesticks, **params)
        save_dataset_as_tfrecords(dataset, dataset_dir, split_ratio=split_ratio, dataset_length=dataset_length, verbose=False)
    ''' stock day 2015-2017 label distribution
    label -1: 108,339
    label 0: 593,256
    label 1: 110,377
    '''


    # trainset = load_dataset(dataset_dir / 'train')
    # validset = load_dataset(dataset_dir / 'valid')
    # #
    # visualize_label_distribution(trainset)
    # visualize_label_distribution(validset)
