from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from QUANTAXIS import QA_Performance, MARKET_TYPE, QA_User, ORDER_DIRECTION
from matplotlib import pyplot as plt
from tqdm import tqdm

from backtest.strategy import NeuralQuantStrategy
from backtest.tools import evaluate_pnl
from settings import OUTPUT_DIR, CRYPTO_TEST_CREDENTIALS
from utilities.visualizations import display_trading_data


class BasePaperTrader:
    def init_account(self, init_cash, credentials, market_type=MARKET_TYPE.CRYPTOCURRENCY):
        username        = credentials['username']
        pwd             = credentials['pwd']
        portfolio_name  = credentials['portfolio_name']
        account_name    = credentials['account_name']

        self.user = QA_User(username=username, password=pwd)
        account = self.user.get_account(portfolio_cookie=portfolio_name, account_cookie=account_name)
        if account is None:
            portfolio = self.user.get_portfolio(portfolio_cookie=portfolio_name)
            account = portfolio.new_account(
                account_cookie=account_name,
                market_type=market_type,
                init_cash=init_cash,
            )
        self.account = account

    def __init__(self, init_cash=10_000, credentials=CRYPTO_TEST_CREDENTIALS, market_type=MARKET_TYPE.CRYPTOCURRENCY):
        self.init_account(init_cash=init_cash, credentials=credentials, market_type=market_type)

    def buy(self, asset_code, price, time, amount: float=None):
        if amount is None:
            if self.account.market_type == MARKET_TYPE.STOCK_CN:
                amount = self.account.cash_available / price // 100 * 100
            if self.account.market_type == MARKET_TYPE.CRYPTOCURRENCY:
                amount = self.account.cash_available // price

        result = self.account.receive_simpledeal(
            code=asset_code,
            trade_price=price,
            trade_amount=amount,
            trade_towards=ORDER_DIRECTION.BUY,
            trade_time=time
        )
        if result != -1:
            print(f'[SIGNAL] 购入{asset_code} 单价：{price:,.2f} 数量：{amount:.4f} 余额: {self.account.cash_available:,.2f}')
        return result == 0

    def sell(self, asset_code, price, time, amount: float=None):
        available = self.account.hold_available.get(asset_code)
        if available is None:
            return
        if amount is None:
            amount = available
        amount = min(amount, available)
        result = self.account.receive_simpledeal(
            code=asset_code,
            trade_price=price,
            trade_amount=amount,
            trade_towards=ORDER_DIRECTION.SELL,
            trade_time=time
        )
        if result != -1:
            print(f'[SIGNAL] 卖出{asset_code} 单价：{price:,.2f} 数量：{amount:.4f} 余额: {self.account.cash_available:,.2f}')
        return result == 0


def paper_trade(symbol, data, trader, weight_file, history_size, future_size, strategy_params=None):
    strategy = NeuralQuantStrategy(weight_file, params=strategy_params)
    print(f"「{strategy.name_cn}」")
    print(f"预处理数据...")
    data = strategy.construct_data(data).dropna()
    print(f"开始纸上交易回测: {symbol}")

    window_size = history_size + future_size
    previous_action = None
    length = len(data) - window_size + 1
    for i in tqdm(range(0, length), total=length, disable=False):
        segment = data.iloc[i: i + window_size].copy()
        bar = segment.iloc[history_size - 1]
        time = segment.index[history_size - 1] \
            if isinstance(segment.index[history_size], pd.Timestamp) \
            else segment.index[history_size - 1][0]

        disp = time > datetime.strptime("2021-07-20", '%Y-%m-%d')

        action = strategy.on_data(segment, symbol, previous_action, history_size=history_size, display=False)
        if trader.account.hold_available.get(symbol) is None:
            if action == 'BUY':
                all_in = trader.account.cash_available // (bar.close * (1 + 1e-3)) - 1
                success = trader.buy(asset_code=symbol, price=bar.close, amount=all_in, time=time)
                if not success:
                    print(all_in, bar.close, trader.account.cash_available)
                previous_action = action
        else:
            if action == 'SELL':
                trader.sell(asset_code=symbol, price=bar.close, time=time)
                previous_action = None

    print(f"回测结束, 账户金额：{trader.account.cash_available:,.2f}")
    print(f"持仓: {trader.account.hold_available}")
    performance = QA_Performance(trader.account)
    pnl_df = performance.pnl
    assert isinstance(pnl_df, pd.DataFrame)
    return pnl_df


def inspect_report(file_name: str=None, pattern: str=None, commission_rate=1e-3, tax_rate=1e-3, display=True):
    import QUANTAXIS as qa

    if file_name is not None:
        if not Path(file_name).exists():
            file_path = OUTPUT_DIR / 'backtests' / file_name
        else:
            file_path = file_name
    else:
        file_path = list(OUTPUT_DIR.rglob(pattern))[0]
    pnl = pd.read_csv(str(file_path))
    report = evaluate_pnl(pnl, commission_rate, tax_rate)
    for key, val in report.items():
        if key.startswith('理论'):
            continue
        if key.endswith('率'):
            val = f"{val * 100:.2f}%"
        if isinstance(val, np.float):
            val = f"{val:.2f}"
        print(f"{key}: {val}")
    print('===\n')

    if not display:
        return

    returns = pnl['true_ratio']
    max_return = returns.max()
    min_return = returns.min()
    returns.plot.hist(bins=len(returns), xlim=[min_return * 2, max_return * 2], title='收益率分布')
    plt.show()

    for i, (index, row) in enumerate(pnl.iterrows()):
        opendate = datetime.strptime(row.opendate, '%Y-%m-%d %H:%M:%S')
        closedate = datetime.strptime(row.closedate, '%Y-%m-%d %H:%M:%S')

        if isinstance(row.code, int):
            symbol = f"{row.code:06d}"
            start_date = opendate - timedelta(days=120)
            end_date = closedate + timedelta(days=20)
            data = qa.QA_fetch_stock_day_adv(symbol, start=start_date, end=end_date).to_qfq().data
        else:
            symbol = row.code
            start_date = opendate - timedelta(days=5)
            end_date = closedate + timedelta(days=5)
            data = qa.QA_fetch_cryptocurrency_min_adv(symbol, str(start_date), str(end_date), frequence='60min').data.dropna()
        # data = indicator_MA(data, 5, 10)
        # data = indicator_MACD(data, )
        # data = indicator_KDJ(data)

        title = f"{closedate - opendate} 收益率: {row.true_ratio * 100:.2f}%"
        display_trading_data(data, pointers=[opendate, closedate], title=title)