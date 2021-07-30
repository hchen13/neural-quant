import sys
from pathlib import Path

import QUANTAXIS as qa

_project_root = Path().cwd().parent
sys.path.append(str(_project_root))

from dataset_management.tools import construct_training_data, load_dataset, visualize_label_distribution, \
    save_dataset_as_tfrecords, fetch_market_caps

if __name__ == '__main__':
    freq = '60min'
    start_date = "2017-01-01"
    end_date = "2020-12-31"

    dataset_dir = Path().home() / 'datasets' / 'quant' / f'crypto{freq}_ATR'
    split_ratio = .05

    params = dict(
        n=144,
        t=6,
        k_up=1.,
        norm=True,
        delta_type='ATR'
    )

    top_n = 20
    market_caps = fetch_market_caps(sort=True)
    coins = [s['symbol'] for s in market_caps[:top_n]]
    for coin in coins:
        symbol = f'BINANCE.{coin}'
        print(f"[INFO] Loading {symbol}@{freq} data: {start_date} ~ {end_date}...")
        data = qa.QA_fetch_cryptocurrency_min_adv(symbol, start=start_date, end=end_date, frequence=freq).data.dropna()
        dataset = construct_training_data(data, **params)
        dataset_length = len(data) - (params['n'] + params['t']) + 1
        print(f"[INFO] creating tfrec files", flush=True)
        save_dataset_as_tfrecords(
            dataset, dataset_dir,
            split_ratio=split_ratio,
            dataset_length=dataset_length,
            buffer_size=1024)
    """
300708it [01:51, 2700.50it/s]
label -1: 50,253
label 0: 195,366
label 1: 55,089
15836it [00:07, 2104.82it/s]
label -1: 2,648
label 0: 10,287
label 1: 2,901
    """

    # start_date = '2012-01-01'
    # end_date = '2018-12-31'
    # params = dict(
    #     n=120,
    #     t=10,
    #     k_up=1.,
    #     norm=True
    # )
    # dataset_dir = Path().home() / 'datasets' / 'quant' / f'stock1d'
    # split_ratio = .001
    # stock_list = qa.QA_fetch_stock_list_adv()
    # for i, (_, row) in tqdm(enumerate(stock_list.iterrows()), total=len(stock_list), disable=False):
    #     symbol = row['code']
    #     queryset = qa.QA_fetch_stock_day_adv(symbol, start=start_date, end=end_date)
    #     if queryset is None or len(queryset) < (params['n'] + params['t']):
    #         continue
    #     # print(f"({i + 1}/{len(stock_list)}): {row['name']}({symbol:s})", end='')
    #     candlesticks = queryset.to_qfq().data.dropna()
    #     dataset_length = len(candlesticks) - (params['n'] + params['t']) + 1
    #     dataset = construct_training_data(candlesticks, **params, shuffle=True)
    #     save_dataset_as_tfrecords(
    #         dataset, dataset_dir,
    #         split_ratio=split_ratio,
    #         dataset_length=dataset_length,
    #         verbose=False, buffer_size=1024)

    trainset = load_dataset(dataset_dir / 'train')
    validset = load_dataset(dataset_dir / 'valid')

    visualize_label_distribution(trainset)
    visualize_label_distribution(validset)


'''
A股分布：
label -1: 608,183
label 0: 2,655,274
label 1: 678,669
'''