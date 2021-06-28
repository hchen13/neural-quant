import hashlib
import re
from datetime import datetime, timedelta

import numpy as np
import QUANTAXIS as qa
import pandas as pd
from scipy.stats import linregress

from settings import MARKET_OPEN_TIME, MARKET_CLOSE_TIME


def create_file_name():
    ts_str = str(datetime.now().timestamp())
    basename = hashlib.sha1(ts_str.encode()).hexdigest()
    return basename


def get_trend(sequence):
    x = np.linspace(0, len(sequence), num=len(sequence))
    slope, intercept, r_val, p_val, stderr = linregress(x.astype('float'), sequence)
    return slope


def get_datetime_name(df):
    cols = ['date', 'datetime']
    for name in cols:
        if name in df.index.names:
            return name
    for name in cols:
        if name in df.keys():
            return name
    raise ValueError("Dataframe does not have any datetime-related indices or columns")


def is_trading_hour(dt: datetime):
    assert isinstance(dt, datetime)
    time = dt.strftime('%H:%M%S')
    return MARKET_OPEN_TIME <= time <= MARKET_CLOSE_TIME


def str2datetime(time_str: str):
    patterns = [
        '%Y-%m-%d',
        '%Y-%m-%d %H:%M',
        '%Y-%m-%d %H:%M:%S'
    ]
    if len(time_str) == 10:
        return datetime.strptime(time_str, patterns[0])
    if len(time_str) == 16:
        return datetime.strptime(time_str, patterns[1])
    else:
        return datetime.strptime(time_str[0:19], patterns[2])


def get_time_duration(start, end, freq='1d'):
    freq = freq.lower()
    assert freq in ['1d', '1m', '5m', '15m', '30m', '60m']
    delta = timedelta(days=1) if freq == '1d' else timedelta(minutes=int(freq[:-1]))
    if isinstance(start, str):
        start = str2datetime(start)
    if isinstance(end, str):
        end = str2datetime(end)
    times = []
    while start < end:
        times.append(start)
        start += delta
    return times


def get_date_duration(start, end, freq='1d'):
    pattern = re.compile("\d{4}-\d{2}-\d{2}")
    if isinstance(start, str):
        assert pattern.match(start)
        start = datetime.strptime(start, '%Y-%m-%d')
    if isinstance(end, str):
        assert pattern.match(end)
        end = datetime.strptime(end, '%Y-%m-%d')
    dates = []
    if freq.endswith('d') or freq.endswith('day'):
        d = timedelta(days=1)
    else:
        minutes = int(freq[:-1])
        d = timedelta(minutes=minutes)
    while start < end:
        dates.append(start)
        start += d
    return dates

def indicator_MA(raw, *ma_list, type='MA'):
    data = raw.copy()
    type = type.upper()
    if type not in ['MA', 'SMA', 'EMA']:
        raise KeyError(f"Invalid value for type: {type}")
    type = 'MA' if type == 'SMA' else type
    func = qa.MA if type == 'MA' else qa.EMA
    for k in ma_list:
        ma_col = f"{type}{k}"
        ma_data = func(data['close'], k)
        data[ma_col] = ma_data
    return data


def indicator_MACD(data, fast=8, slow=21, mid=5):
    df = qa.MACD(data.close, FAST=fast, SLOW=slow, MID=mid)
    df.rename(columns={
        'DIFF': 'MACD_DIFF',
        'DEA': 'MACD_DEA',
        'MACD': 'MACD_MACD'
    }, inplace=True)
    if 'MACD_DIFF' in data.keys():
        data = data.drop(columns=['MACD_DIFF', 'MACD_DEA', 'MACD_MACD'])
    data = pd.concat([data, df], axis=1)
    return data


def indicator_KDJ(data, n=9, m1=3, m2=3):
    df = qa.QA_indicator_KDJ(data, n, m1, m2)
    if 'KDJ_K' in data.keys():
        data = data.drop(columns=['KDJ_K', 'KDJ_D', 'KDJ_J'])
    data = pd.concat([data, df], axis=1)
    return data


def indicator_RSI(data, n1=12, n2=26, n3=9):
    df = qa.QA_indicator_RSI(data, n1, n2, n3)
    if 'RSI1' in data.keys():
        data = data.drop(columns=['RSI1', 'RSI2', 'RSI3'])
    data = pd.concat([data, df], axis=1)
    return data


def indicator_ATR(data, n=14):
    df = qa.QA_indicator_ATR(data, n).drop(columns=['TR'])
    data = pd.concat([data, df], axis=1)
    return data


def indicator_KeltnerBand(data, n_atr=14, n_ema=20, n=2):
    """ 基于ATR的Keltner Band(Keltner Channel)
    n_atr: ATR周期
    n_ema: EMA周期
    n: 通道上下界ATR倍数
    """
    atr = qa.QA_indicator_ATR(data, n_atr)
    ema = qa.EMA(data['close'], n_ema)
    data['Keltner_UB'] = ema + n * atr['ATR']
    data['Keltner_MID'] = ema
    data['Keltner_LB'] = ema - n * atr['ATR']
    return data
