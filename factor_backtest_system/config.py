"""
config.py
回测系统全局参数配置
"""
import pandas as pd

# ========== 交易成本参数 ==========

# 佣金 (双边, 买卖各收一次)
COMMISSION_RATE = 0.0001  # 万1

# 印花税 (仅卖出时收取)
# 2023-08-28 之前: 0.1%, 之后: 0.05%
STAMP_TAX_DATE = pd.Timestamp('2023-08-28')
STAMP_TAX_BEFORE = 0.001   # 0.1%
STAMP_TAX_AFTER = 0.0005   # 0.05%

# 滑点 (单边)
SLIPPAGE_RATE = 0.001  # 0.1%


def get_buy_cost():
    """买入单边成本 = 佣金 + 滑点"""
    return COMMISSION_RATE + SLIPPAGE_RATE


def get_sell_cost(date):
    """
    卖出单边成本 = 佣金 + 印花税 + 滑点
    印花税根据日期不同而不同
    """
    if date < STAMP_TAX_DATE:
        stamp_tax = STAMP_TAX_BEFORE
    else:
        stamp_tax = STAMP_TAX_AFTER
    return COMMISSION_RATE + stamp_tax + SLIPPAGE_RATE
