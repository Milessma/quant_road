import pandas as pd
import numpy as np
from tqdm import tqdm
from numpy.lib.stride_tricks import sliding_window_view
from factor_backtest_system.backtest_utils import *
from common_utils import log

def STR_FACTOR(start_date, end_date, sita=0.1, delta=0.7, window_size=21):
    # 1. 计算回溯起始日期 (提前约30-40天以确保交易日充足)
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=45)).strftime('%Y-%m-%d')
    
    start_year = pd.to_datetime(lookback_start).year
    end_year = pd.to_datetime(end_date).year

    # 2. 加载数据
    log("======= 加载数据并计算收益率、市场中位数收益率 =======")
    df_daily = load_parquet_by_years('daily', 'daily', start_year, end_year)
    
    # 3. 计算基础收益率
    df_return = pd.DataFrame(index=df_daily.index)
    # 按 order_book_id 分组后对 close 进行 shift(1) 计算收益率
    df_return['return'] = df_daily.groupby(level='order_book_id')['close'].pct_change()

    # 4. 计算市场中位数收益率
    market_return = df_return['return'].groupby(level='date').median()
    df_return['market_return'] = df_return.index.get_level_values('date').map(market_return)

    # 5. 计算每日显著性得分 sigma
    log("======= 计算每日显著性得分 =======")
    diff = (df_return['return'] - df_return['market_return']).abs()
    norm = df_return['return'].abs() + df_return['market_return'].abs() + sita
    df_return['sigma'] = diff / norm

    # 6. 预计算权重向量
    weights_fixed = delta ** (np.arange(window_size))
    weights_fixed /= weights_fixed.sum()

    # 7. 高性能滚动计算函数 (矩阵化)
    def fast_rolling_str(group):
        rets = group['return'].values
        sigmas = group['sigma'].values
        n = len(rets)
        
        if n < window_size:
            return pd.Series(np.nan, index=group.index)

        # 构造滑动窗口矩阵
        w_rets = sliding_window_view(rets, window_size)
        w_sigmas = sliding_window_view(sigmas, window_size)
        
        # 截面排序索引 (降序)
        sort_indices = np.argsort(-w_sigmas, axis=1)
        
        # 提取排序后的收益率并进行权重内积
        row_indices = np.arange(len(w_rets))[:, None]
        sorted_w_rets = w_rets[row_indices, sort_indices]
        str_values = sorted_w_rets @ weights_fixed
        
        # 拼接结果
        full_res = np.concatenate([np.full(window_size - 1, np.nan), str_values])
        return pd.Series(full_res, index=group.index)

    # 8. 分组并行/应用计算
    log("======= 计算 STR 因子 ========")
    tqdm.pandas(desc="Calculating STR Factor")
    df_return['str_factor'] = df_return.groupby(level='order_book_id', group_keys=False).progress_apply(fast_rolling_str)

    # 9. 截取用户请求的时间段
    final_factor = df_return.loc[pd.IndexSlice[:, start_date:end_date], :]
    
    return final_factor.sort_index()

if __name__ == '__main__':
    str = STR_FACTOR('2016-01-01', '2025-12-31')
    print(1)