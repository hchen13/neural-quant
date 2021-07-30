from datetime import datetime, timedelta

import pandas as pd
from QUANTAXIS import MARKET_TYPE, QA_Risk, QA_fetch_cryptocurrency_day_adv, QA_util_date_stamp
from QUANTAXIS.QAUtil.QADate_Adv import QA_util_timestamp_to_str
from dateutil.tz import tzutc
from matplotlib import pyplot as plt

from backtest.papertrade import BasePaperTrader
from settings import BACKTEST_DIR, CRYPTO_TEST_CREDENTIALS


pd.set_option('display.max_columns', None)


if __name__ == '__main__':
    file = BACKTEST_DIR / 'BINANCE.ETHUSDT60min.csv'
    table = pd.read_csv(str(file))
    opendate = table['opendate'].apply(lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
    closedate = table['closedate'].apply(lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
    start_time = opendate.min()
    end_time = closedate.max()
    trader = BasePaperTrader(
        init_cash=1_000_000,
        credentials=CRYPTO_TEST_CREDENTIALS,
        market_type=MARKET_TYPE.CRYPTOCURRENCY,
        commission_rate=1e-3,
        tax_rate=0
    )
    for index, row in table.iterrows():
        opendate = row['opendate']
        closedate = row['closedate']
        symbol = row['code']
        buy_price = row['buy_price']
        amount = row['amount']
        sell_price = row['sell_price']
        trader.buy(asset_code=symbol, price=buy_price, time=opendate, amount=amount, verbose=False)
        trader.sell(asset_code=symbol, price=sell_price, time=closedate, verbose=False)

    start_date = datetime.strptime(trader.account.start_date, "%Y-%m-%d") + timedelta(hours=8)
    end_date = datetime.strptime(trader.account.end_date, "%Y-%m-%d") + timedelta(hours=24)
    market_data = QA_fetch_cryptocurrency_day_adv('BINANCE.ETHUSDT', start=start_date, end=end_date)
    risk = QA_Risk(
        trader.account,
        benchmark_code='BINANCE.BTCUSDT', benchmark_type=MARKET_TYPE.CRYPTOCURRENCY,
        market_data=market_data
    )
    # risk.assets.plot()
    # plt.show()
    # print(risk.account.daily_hold)
    # print(risk.market_data.data)
    # plt.show()
