from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT        = Path(__file__).parent
OUTPUT_DIR          = PROJECT_ROOT / 'outputs'
WEIGHTS_DIR         = OUTPUT_DIR / 'weights'
BACKTEST_DIR        = OUTPUT_DIR / 'backtests'

MA_COLORS           = ['white', 'yellow', 'magenta', 'lightgreen', 'lightgray', 'blue']

MARKET_OPEN_TIME    = '09:30:00'
MARKET_CLOSE_TIME   = "15:00:00"

# TUSHARE_TOKEN       = os.getenv('TOKEN')
# QASETTING.set_config('TSPRO', 'token', TUSHARE_TOKEN)

BACKTEST_DIR.mkdir(exist_ok=True, parents=True)
WEIGHTS_DIR.mkdir(exist_ok=True, parents=True)



DEFAULT_CRYPTO_MARKET   = 'BINANCE'


# backtest related
CRYPTO_TEST_CREDENTIALS = dict(
    username       = 'ethan_test_0',
    pwd            = 'qwer',
    portfolio_name = 'portfolio_1',
    account_name   = 'crypto_0',
)