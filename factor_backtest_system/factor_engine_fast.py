"""
factor_engine_fast.py
向量化加速版 calc_quantile_stats + run_factor_test
逻辑与 factor_engine.py 完全一致, 只是用 numpy 矩阵运算替代 Python for 循环
"""
import pandas as pd
import numpy as np
import tqdm
from backtest_utils import *
from common_utils import *

# 复用原文件的绑定函数 (绘图、预处理、IC等)
from factor_engine import (
    plot_factor_stats,
    plot_factor_coverage,
    plot_ic_cumulative,
    plot_ic_yearly,
    plot_quantile_returns,
    plot_long_short,
    _calc_drawdown,
    get_future_returns,
    preprocess_factor,
    calc_ic_stats,
)


def calc_quantile_stats_fast(wide_factor, wide_close, wide_can_buy, wide_can_sell, bins=5, hold_days=1):
    """
    分层收益计算 (向量化加速版)
    
    逻辑与 factor_engine.calc_quantile_stats 完全一致:
    - T-1日因子分组 → T日买入 → T+hold_days日卖出
    - wide_can_buy: T日是否能买
    - wide_can_sell: T日是否能卖
    - 卖不出的股票被迫继续持有, 其收益仍计入组合
    - wide_ret_daily: 日频收益 (N=1), 用于逐日计算组合收益
    - hold_days: 调仓周期, 每 hold_days 天重新分组一次
    
    加速方式: 全部转 numpy, 逐日只做矩阵运算, 消除内层 Python 循环
    """
    # 1. 只对可买入的股票进行分组
    buyable_factor = wide_factor.where(wide_can_buy)
    wide_ranks = buyable_factor.rank(axis=1, pct=True) # .rank(axis=1, pct=True): 在行上计算百分比排名, 输出为(0, 1]
    wide_quantiles = (wide_ranks * bins).apply(np.ceil) # * bins: 线性缩放, 将排名映射到(0, bins] 区间, .apply(np.ceil): 向上取整, 将连续数值转换为{1, 2, ..., bins}的整数标签
    
    # 2. 转 numpy (T x N)
    q_arr = wide_quantiles.values                    # float, NaN = 无分组
    close_arr = wide_close.values                    # (T, N) 收盘价矩阵
    can_sell_arr = wide_can_sell.fillna(True).values.astype(bool)  # NaN 视为可卖
    
    T, N = q_arr.shape # q_arr index 是日期 columns 是股票 values 是组别
    
    # 3. 预计算每组的 target mask: (bins, T, N) bool
    #    target_masks[b-1, t, :] = (q_arr[t, :] == b)
    target_masks = np.zeros((bins, T, N), dtype=bool)
    for b in range(1, bins + 1):
        target_masks[b - 1] = (q_arr == b)
    
    # 4. 逐日模拟, 但内层全向量化
    # prev_hold: (bins, N) bool — 上期各组持仓
    prev_hold = np.zeros((bins, N), dtype=bool)
    # group_rets: (T, bins)
    group_rets = np.zeros((T, bins), dtype=np.float64)
    # price_base[b, i] = 调仓日时股票 i 的收盘价 (用于计算累计收益)
    price_base = np.full((bins, N), np.nan, dtype=np.float64)
    # 净值
    base_nav = np.ones(bins) # 调仓日开始时的基准净值

    for t in tqdm.tqdm(range(T), desc='分层持仓模拟(fast)'):
        sell_ok = can_sell_arr[t]          # (N,) 今天能否卖出
        # 判断是否为调仓日: 从 t=1 开始, 每 hold_days 天调一次仓 因为T日要依据T-1日的因子
        is_rebalance = (t >= 1) and ((t - 1) % hold_days == 0)
        for b in range(bins):
            if is_rebalance:
                if t > 0:
                    base_nav[b] = group_rets[t-1, b]
                target = target_masks[b, t]    # (N,) 调仓日: 用新因子分组
            else:
                target = prev_hold[b]          # (N,) 非调仓日: 沿用上期持仓
            
            forced = prev_hold[b] & ~sell_ok  # (N,) 上期持有 且 今天卖不出
            actual = target | forced       # (N,) 实际持仓 = 目标持仓 + 强制持仓

            if is_rebalance:
                # 记录调仓日的买入价格
                # np.where(条件, if True取这个值, if False取这个值)
                price_base[b] = np.where(actual, close_arr[t], np.nan) # 如果一只股票在上期被强制持有 (forced) 不在新的 target 里，但它的 price_base 应该保留之前的值 而不是被清空 但是由于此时这个股票 权重不一 不该等权 

            cum_ret = (close_arr[t] - price_base[b]) / price_base[b] # (N, ) 累计收益率
            mask = actual & np.isfinite(cum_ret) # 创建 bool 数组，筛选出没问题的股票
            cnt = mask.sum() # 一共有多少只股票
            period_ret = cum_ret[mask].sum() / cnt if cnt > 0 else 0.0
            # t = 0 时 group_rets[0, b] = 1
            group_rets[t, b] = base_nav[b] * (1 + period_ret)
            prev_hold[b] = actual
    cols = [f'Group_{b}' for b in range(1, bins + 1)]
    return pd.DataFrame(group_rets, index=wide_factor.index, columns=cols)


def run_factor_test(df_factor, start_date, end_date, factor_col='str_factor', bins=5, N=1):
    """
    一键运行全流程 (加速版)
    时序: T-1日因子 → T日买入 → T+N日卖出
    """
    # --- 1. 数据准备 (长转宽) ---
    log("======= 数据准备(含交易约束) =======")
    # df_daily = get_K_data(start_date, end_date)
    df_daily = pd.read_parquet("G:\quant_road\K_clean_data.parquet")
    wide_factor = df_factor[factor_col].unstack(level='order_book_id')
    wide_close = df_daily['close'].unstack(level='order_book_id')
    wide_can_buy = df_daily['can_buy'].unstack(level='order_book_id')
    wide_can_sell = df_daily['can_sell'].unstack(level='order_book_id')
    
    # 对齐索引
    common_dates = wide_factor.index.intersection(wide_close.index)
    wide_factor = wide_factor.loc[common_dates]
    wide_close = wide_close.loc[common_dates]
    wide_can_buy = wide_can_buy.loc[common_dates]
    wide_can_sell = wide_can_sell.loc[common_dates]

    # 新视角: T日为交易日, 因子来自T-1日
    # 因子 shift(1): T-1日因子对齐到T日行
    # can_buy / can_sell: 直接用T日的, 不需要shift
    wide_factor = wide_factor.shift(1)

    # --- 2. 预处理 ---
    log("======= 因子预处理 =======")
    processed_factor = preprocess_factor(wide_factor)

    # --- 3. 计算收益率 ---
    log("======= 收益率计算 =======")
    wide_ret = get_future_returns(wide_close, N=N)        # N天总收益, 用于IC

    # --- 4. 评价 ---
    log("======= IC分析 =======")
    ic_df = calc_ic_stats(processed_factor, wide_ret)

    log("======= 分层汇总(含交易约束, 加速版) =======")
    quantile_df = calc_quantile_stats_fast(
        processed_factor, wide_close, wide_can_buy, wide_can_sell, bins=bins, hold_days=N
    )
    
    # --- 5. 可视化 ---
    log("======= 绘制IC累积图 =======")
    plot_ic_cumulative(ic_df, factor_col=factor_col)
    log("======= 绘制分年度IC =======")
    plot_ic_yearly(ic_df, factor_col=factor_col)
    log("======= 绘制分层收益图 =======")
    plot_quantile_returns(quantile_df, bins=bins, factor_col=factor_col)
    log("======= 绘制多头/多空净值与回撤 =======")
    plot_long_short(quantile_df, bins=bins, factor_col=factor_col)

    # --- 6. 汇总结果 ---
    results = {
        'ic_df': ic_df,
        'quantile_df': quantile_df,
        'summary': {
            'IC_Mean': ic_df['IC'].mean(),
            'Rank_IC_Mean': ic_df['Rank_IC'].mean(),
            'IC_IR': ic_df['IC'].mean() / ic_df['IC'].std() if ic_df['IC'].std() != 0 else 0,
            'Long_Short_Ret': (quantile_df[f'Group_{bins}'] - quantile_df['Group_1']).mean()
        }
    }
    
    log("======= 结果汇总 =======")
    log(f"IC_Mean: {ic_df['IC'].mean()}")
    log(f"Rank_IC_Mean: {ic_df['Rank_IC'].mean()}")
    log(f"IC_IR: {ic_df['IC'].mean() / ic_df['IC'].std() if ic_df['IC'].std() != 0 else 0}")
    log(f"Long_Short_Ret: {((quantile_df[f'Group_{bins}'] - quantile_df['Group_1']).mean())*100}%")
    
    return results


if __name__ == '__main__':
    period = 60
    df_factor = pd.read_parquet(r'G:\quant_road\str.parquet')
    run_factor_test(df_factor, '2016-01-01', '2025-12-31', bins=10, N=period)
