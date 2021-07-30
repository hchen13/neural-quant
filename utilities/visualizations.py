import re
from datetime import datetime

import matplotlib
import matplotlib.dates as mpl_dates
import mplfinance as mpf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from settings import MA_COLORS
# matplotlib.rcParams['font.family'] = 'FZRuiZhengHei_GBK'
from utilities.funcs import get_datetime_name

matplotlib.rcParams['font.family'] = 'PingFang HK'
matplotlib.rcParams['axes.unicode_minus'] = False


def set_style(*axes):
    for axis in axes:
        axis.set_facecolor((46 / 255, 52 / 255, 70 / 255))
        axis.grid(color=(1, 1, 1, .1), which='major', linewidth='.5', linestyle='-', alpha=.1)
        axis.xaxis.label.set_color('white')
        axis.yaxis.label.set_color('white')
        axis.tick_params(colors='white', which='both')
        axis.xaxis.set_major_formatter(mpl_dates.DateFormatter('%Y-%-m-%-d'))


def mpf_draw_MACD(chunk, axis):
    dif_line = mpf.make_addplot(chunk['MACD_DIFF'], type='line', color='white', width=.5, ax=axis)
    dea_line = mpf.make_addplot(chunk['MACD_DEA'], type='line', color='yellow', width=.5, ax=axis)
    colors = ['red' if k > 0 else 'green' for k in chunk['MACD_MACD']]
    hist_bar = mpf.make_addplot(chunk['MACD_MACD'], type='bar', color=colors, ax=axis)
    return [dif_line, dea_line, hist_bar]


def mpf_draw_KDJ(chunk, axis, supports=(20, 80)):
    axis.set_ylabel('KDJ')

    k = mpf.make_addplot(chunk['KDJ_K'], type='line', color='white', width=.5, ax=axis)
    d = mpf.make_addplot(chunk['KDJ_D'], type='line', color='yellow', width=.5, ax=axis)
    j = mpf.make_addplot(chunk['KDJ_J'], type='line', color='magenta', width=.5, ax=axis)
    if supports:
        for y in supports:
            color = 'g' if y < 50 else 'r'
            axis.axhline(y, color=color, linewidth=.5, ls='dotted')
    return [k, d, j]


def mpf_draw_RSI(chunk, axis, supports=(30, 70)):
    axis.set_ylabel('RSI')
    l1 = mpf.make_addplot(chunk['RSI1'], type='line', color='white', width=.5, ax=axis)
    l2 = mpf.make_addplot(chunk['RSI2'], type='line', color='yellow', width=.5, ax=axis)
    l3 = mpf.make_addplot(chunk['RSI3'], type='line', color='magenta', width=.5, ax=axis)
    if supports:
        for y in supports:
            color = 'g' if y < 50 else 'r'
            axis.axhline(y, color=color, linewidth=.5, ls=':')
    axis.axhline(50, color='#ddd', linewidth=.5, ls='--')
    return [l1, l2, l3]


def mpf_draw_ATR(chunk, axis):
    axis.set_ylabel('ATR')
    atr_line = mpf.make_addplot(chunk['ATR'], type='line', color='white', width=.5, ax=axis)
    return [atr_line]


def draw_trading_data(data:pd.DataFrame, pointers=None, title=None, figsize=(12, 8)):
    chunk = data.copy()
    date_col = get_datetime_name(chunk)
    chunk.reset_index(inplace=True)
    chunk.set_index(date_col, inplace=True)
    indicators = []
    for ind in ['MACD', 'KDJ', 'RSI', 'ATR']:
        for key in chunk.keys():
            if ind in key:
                indicators.append(ind)
                break

    nrows = 2
    height_ratios = [2, .5]
    for _ in indicators:
        nrows += 1
        height_ratios.append(1)

    fig, axes = plt.subplots(
        figsize=figsize,
        nrows=nrows, sharex='all',
        gridspec_kw={
            'height_ratios': height_ratios
        }
    )
    fig.subplots_adjust(hspace=0)
    fig.set_facecolor((35 / 255, 42 / 255, 60 / 255))
    set_style(*axes)
    main_ax, vol_ax = axes[:2]
    subplots = []

    # MA or SMA
    ma_pattern = re.compile("^MA\d+")
    ma_cols = [key for key in chunk.keys() if ma_pattern.match(key)]
    for ma_col, color in zip(ma_cols, MA_COLORS):
        ap = mpf.make_addplot(chunk[ma_col], type='line', ax=main_ax, color=color, width=.5)
        subplots.append(ap)

    ema_pattern = re.compile("^EMA\d+")
    ema_cols = [key for key in chunk.keys() if ema_pattern.match(key)]
    for col, color in zip(ema_cols, MA_COLORS):
        ap = mpf.make_addplot(chunk[col], type='line', ax=main_ax, color=color, width=.5)
        subplots.append(ap)

    # Tunnels 通道
    tunnels = ['Keltner', "Boll"]
    tunnel_ub, tunnel_lb = None, None
    for tn in tunnels:
        ub_col, m_col, lb_col = f"{tn}_UB", f"{tn}_MID", f"{tn}_LB"
        if m_col in chunk.keys():
            for col in [ub_col, m_col, lb_col]:
                ap = mpf.make_addplot(chunk[col], type='line', ax=main_ax, color='goldenrod', width=.5)
                subplots.append(ap)
            tunnel_ub = chunk[ub_col].values
            tunnel_lb = chunk[lb_col].values

    # indicators
    for i, ind in enumerate(indicators):
        func_name = f'mpf_draw_{ind}'
        aps = globals()[func_name](chunk, axes[i + 2])
        subplots += aps

    # pointers
    if pointers is None:
        pointers = []
    assert isinstance(pointers, list)
    signals = []
    last = None
    for i in chunk.index:
        found = False
        for pt in pointers:
            if isinstance(pt, str):
                pt = datetime.strptime(pt, "%Y-%m-%d %H:%M:%S")
            if i > pt and last <= pt:
                signals.append(chunk.loc[last]['low'])
                found = True
                break
        if not found:
            signals.append(np.nan)
        last = i
    subplots.append(mpf.make_addplot(signals, type='scatter', markersize=100, marker='^', ax=main_ax, color='orange'))

    if title is None:
        d0 = str(chunk.index[0])
        dt = str(chunk.index[-1])
        # code = chunk.iloc[0].code
        title = f"{d0} - {dt}"

    fig.suptitle(title, color='w')
    mpf.plot(chunk, type='candle', addplot=subplots, style='yahoo', ax=main_ax, volume=vol_ax)

    if tunnel_ub is not None:
        main_ax.fill_between(x=np.arange(len(tunnel_ub)), y1=tunnel_ub, y2=tunnel_lb, color='goldenrod', alpha=.1)

    main_ax.legend(ma_cols + ema_cols)
    return fig, axes


def display_trading_data(data: pd.DataFrame, pointers=None, title=None, figsize=(12, 8)):
    fig, axes = draw_trading_data(data, pointers, title, figsize)
    plt.draw()
    plt.waitforbuttonpress()
    plt.close()
    return fig, axes
