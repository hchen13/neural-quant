from datetime import datetime

import numpy as np

from utilities.funcs import str2datetime


class Actions:
    NOTHING = 0
    BUY = 1
    SELL = 2


class Action:
    ACTIONS = ['NOTHING', 'BUY', 'SELL']

    def _parse_action(self, action):
        if isinstance(action, int) and 0 <= action <= 2:
            return action
        if isinstance(action, str):
            action = action.upper()
            if action in self.ACTIONS:
                return getattr(Actions, action)
        raise KeyError(f"[Action] unknown action: {action}")

    def __init__(self, action, take_profit=None, stop_loss=None):
        self.action = self._parse_action(action)
        self.take_profit = take_profit
        self.stop_loss = stop_loss
        self._action_str = self.ACTIONS[self.action]

    def __eq__(self, other):
        if isinstance(other, int):
            return self.action == other
        if isinstance(other, str):
            return self._action_str == other.upper()
        return self == other

    def __repr__(self):
        return self._action_str

    def should_close(self, price):
        if self == 'buy':
            if self.stop_loss is not None and price < self.stop_loss:
                return True
            if self.take_profit is not None and price > self.take_profit:
                return True
        if self == 'sell':
            if self.stop_loss is not None and price > self.stop_loss:
                return True
            if self.take_profit is not None and price < self.take_profit:
                return True
        return False


class BaseStrategy:
    name = "Base Strategy"
    name_cn = "基策略"

    def __init__(self):
        pass

    def on_data(self, segment, code, *args, **kwargs):
        """main logic for the strategy implemented here
        """
        raise NotImplementedError("method `on_data(segment, code)` not implemented")

    def construct_data(self, raw_data, *args, **kwargs):
        """Compute and concatenate the indicators needed for the strategy executions
        """
        raise NotImplementedError("method `construct_data(segment, code)` not implemented")


def compute_return(table):
    report = {}
    for col in ['pnl_ratio', 'true_ratio']:
        prefix = '实际' if col.startswith('true_') else "理论"
        returns = table[col]
        total_return = (returns + 1).prod() - 1
        report[f"{prefix}总收益率"] = total_return
    return report


def evaluate_pnl(table, commission_rate, tax_rate):
    symbols = table['code'].unique()
    n_trades = len(table)
    opendate = table['opendate'].apply(lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
    closedate = table['closedate'].apply(lambda s: datetime.strptime(s, "%Y-%m-%d %H:%M:%S"))
    start_time = opendate.min()
    end_time = closedate.max()

    duration = end_time - start_time
    duration_years = duration.total_seconds() / 3600 / 24 / 365
    trades_per_year = n_trades / duration_years

    report = {'标的数量': len(symbols)}
    report['交易数量'] = len(table)
    report['起始时间'] = start_time
    report['结束时间'] = end_time
    report['持续时间'] = duration

    d0 = table['opendate'].apply(str2datetime)
    d1 = table['closedate'].apply(str2datetime)
    hold_gap = (d1 - d0).apply(lambda v: v.total_seconds() / 3600 / 24)
    table['gap_day'] = hold_gap
    report['最大持仓天数'] = hold_gap.max()
    report['最小持仓天数'] = hold_gap.min()
    report['平均持仓天数'] = hold_gap.mean()

    returns = table['pnl_ratio'] + 1
    true_ratio = (1 - commission_rate - tax_rate) * returns - commission_rate - 1
    table['true_ratio'] = true_ratio

    tmp_report = {}
    for symbol in symbols:
        tb = table[table['code'] == symbol]
        res = compute_return(tb)
        for key, val in res.items():
            if key in tmp_report.keys():
                tmp_report[key].append(val)
            else:
                tmp_report[key] = [val]

    for col in ['pnl_ratio', 'true_ratio']:
        prefix = '实际' if col.startswith('true_') else "理论"
        returns = table[col]
        wins = table[returns >= 0]
        losses = table[returns < 0]
        total_returns = tmp_report[f"{prefix}总收益率"]
        total_return = np.mean(total_returns)
        monthly_return = (total_return + 1) ** (1 / duration_years / 12) - 1
        annualized_ret = (monthly_return + 1) ** 12 - 1
        volatility = np.std(returns, ddof=0)
        annualized_vol = volatility * (trades_per_year ** .5)
        sharpe_ratio = annualized_ret / (annualized_vol + 1e-8)
        report[f"{prefix}利润单平均持仓时间"] = wins['gap_day'].mean()
        report[f"{prefix}亏损单平均持仓时间"] = losses['gap_day'].mean()
        report[f"{prefix}胜场"] = len(wins)
        report[f"{prefix}负场"] = len(losses)
        report[f"{prefix}胜率"] = len(wins) / n_trades
        report[f"{prefix}最大收益率"] = returns.max()
        report[f"{prefix}最大亏损率"] = returns.min()
        report[f'{prefix}平均收益率'] = returns.mean()
        report[f"{prefix}利润单平均收益率"] = wins[col].mean()
        report[f"{prefix}亏损单平均收益率"] = losses[col].mean()
        report[f"{prefix}总收益率"] = total_return
        report[f"{prefix}月均收益率"] = monthly_return
        report[f"{prefix}年化收益率"] = annualized_ret
        report[f"{prefix}波动率"] = volatility
        report[f"{prefix}年化波动率"] = annualized_vol
        report[f"{prefix}夏普比"] = sharpe_ratio
    return report
