from datetime import datetime

from QUANTAXIS import QA_fetch_cryptocurrency_min_adv, MARKET_TYPE

from backtest.papertrade import BasePaperTrader, paper_trade, inspect_report
from settings import CRYPTO_TEST_CREDENTIALS, OUTPUT_DIR


def main():
    symbol = 'BINANCE.ETHUSDT'
    freq = '60min'
    start_date = '2021-01-01'
    end_date = str(datetime.now())
    model_file = 'fcn_crypto_std1.h5'
    history_size, future_size = 144, 12
    strategy_params = {
        'BUY_OPEN_THRESHOLD': .9,
        'SELL_CLOSE_THRESHOLD': None,
        'STOP_LOSS': False
    }

    print(f"[INFO] Fetching {symbol}@{freq} data: {start_date} ~ {end_date}...")
    data = QA_fetch_cryptocurrency_min_adv(symbol, start=start_date, end=end_date, frequence=freq).data.dropna()
    print(f"[INFO] {len(data)} candlesticks fetched.\n")

    trader = BasePaperTrader(
        init_cash=1_000_000,
        credentials=CRYPTO_TEST_CREDENTIALS,
        market_type=MARKET_TYPE.CRYPTOCURRENCY
    )
    pnl = paper_trade(symbol, data, trader, model_file, history_size, future_size, strategy_params=strategy_params)

    save_path = OUTPUT_DIR / 'backtests' / f"{symbol}{freq}.csv"
    print(f"储存回测记录: {str(save_path)}")
    pnl.to_csv(str(save_path))

    inspect_report(file_name=str(save_path), commission_rate=1e-3, tax_rate=0, display=True)


if __name__ == '__main__':
    main()