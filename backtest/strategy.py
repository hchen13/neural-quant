import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from backtest.tools import BaseStrategy, Action
from dataset_management.tools import normalize_ohlcv
from prototype.fcn import build_fcn
from settings import OUTPUT_DIR
from utilities.visualizations import draw_trading_data


class NeuralQuantStrategy(BaseStrategy):
    name = 'Neural Quant TBM Strategy'
    name_cn = '深度量化TBM策略'

    DEFAULT_PARAMS = {
        'BUY_OPEN_THRESHOLD': None,
        'SELL_CLOSE_THRESHOLD': None,
        'STOP_LOSS': True
    }

    def __getattr__(self, item):
        if item in self.params.keys():
            return self.params[item]
        return getattr(self, item)

    def _check_params(self, params):
        for ind in params.keys():
            assert ind in self.DEFAULT_PARAMS.keys()

    def _construct_params(self, params):
        if params is None:
            params = self.DEFAULT_PARAMS
        params = {**self.DEFAULT_PARAMS, **params}
        self._check_params(params)
        return params

    def __init__(self, weight_file, params: dict = None):
        super(NeuralQuantStrategy, self).__init__()
        weight_dir = OUTPUT_DIR / 'weights'
        weight_path = weight_dir / weight_file
        print(f"[NeuralQuantStrategy] Building model and loading weights from {weight_path}")
        self.model = build_fcn(None)
        self.model.load_weights(str(weight_path), by_name=True)
        self.params = self._construct_params(params)

    def construct_data(self, raw_data, *args, **kwargs):
        return raw_data

    def predict(self, history: pd.DataFrame):
        _feed_columns = ['open', 'high', 'low', 'close', 'volume']
        feed = np.expand_dims(history[_feed_columns].values, axis=0).astype('float32')
        pred = self.model.predict(feed)
        label = np.argmax(pred, axis=-1).squeeze() - 1
        confidence = np.max(pred, axis=-1).squeeze()
        return label, confidence

    def should_buy(self, label, confidence):
        decision = label == 1
        if self.BUY_OPEN_THRESHOLD is not None:
            decision = decision and confidence > self.BUY_OPEN_THRESHOLD
        return decision

    def should_sell(self, label, confidence, bar, previous_action: Action):
        if previous_action is None:
            return False
        if self.STOP_LOSS and previous_action.should_close(bar['close']):
            return True
        decision = label == -1
        if self.SELL_CLOSE_THRESHOLD is not None:
            decision = decision and confidence > self.SELL_CLOSE_THRESHOLD
        return decision

    def on_data(self, segment, code, previous_action=None, history_size=None, display=False, *args, **kwargs):
        bar = segment.iloc[history_size - 1]
        norm_data = normalize_ohlcv(segment, history_size)
        history = norm_data.iloc[:history_size]
        time = history.index[-1] if isinstance(history.index[-1], pd.Timestamp) else history.index[-1][0]
        label, confidence = self.predict(history)
        action = Action('nothing')
        if self.should_buy(label, confidence):
            hist = segment.iloc[:history_size]
            volatility = hist['close'].std(ddof=0)
            action = Action('buy', stop_loss=bar['close'] - volatility)
        if self.should_sell(label, confidence, bar, previous_action):
            action = Action('sell')

        if display:
            hist = segment.iloc[:history_size]
            volatility = hist['close'].std(ddof=0)
            k1, k2 = 1., 1.
            top = k1 * volatility + hist['close'].iloc[-1]
            bot = -k2 * volatility + hist['close'].iloc[-1]
            fig, axes = draw_trading_data(segment, pointers=[time],
                                          title=f"label: {label} confidence: {confidence * 100:.2f}% action: {action}")
            main_ax = axes[0]
            main_ax.axhline(top, color='g', linewidth=1.5, ls='--')
            main_ax.axhline(bot, color='r', linewidth=1.5, ls='--')
            plt.show()

        return action