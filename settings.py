from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT        = Path(__file__).parent
OUTPUT_DIR          = PROJECT_ROOT / 'outputs'

MA_COLORS           = ['white', 'yellow', 'magenta', 'lightgreen', 'lightgray', 'blue']

MARKET_OPEN_TIME    = '09:30:00'
MARKET_CLOSE_TIME   = "15:00:00"

# TUSHARE_TOKEN       = os.getenv('TOKEN')
# QASETTING.set_config('TSPRO', 'token', TUSHARE_TOKEN)

OUTPUT_DIR.mkdir(exist_ok=True)


DEFAULT_CRYPTO_MARKET   = 'HUOBI_PRO'