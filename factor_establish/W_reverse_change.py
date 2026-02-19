import pandas as pd
import numpy as np
from factor_backtest_system.backtest_utils import *
from numpy.lib.stride_tricks import sliding_window_view
from factor_backtest_system.factor_engine_fast import *

def M_high_2_0(start_date, end_date, N=10, window_size=20):
    lookback_start = (pd.to_datetime(start_date) - pd.Timedelta(days=40)).strftime('%Y-%m-%d')

    start_year = pd.to_datetime(lookback_start).year
    end_year = pd.to_datetime(end_date).year

    log("======= 加载数据、计算收益率 =======")
    df_daily = load_parquet_by_years('daily', 'daily', start_year, end_year)

    df_return = pd.DataFrame(index=df_daily.index)
    df_return['ret'] = df_daily.groupby(level='order_book_id')['close'].pct_change()

    df_return['avg_trade_turnover'] = df_daily['total_turnover'] / df_daily['num_trades']
    
    def calc_rolling_top_n_sum(group):
        turnover = group['avg_trade_turnover'].fillna(-np.inf).values
        ret = group['ret'].fillna(0).values

        if len(turnover) < window_size:
            return pd.Series(np.nan, index=group.index)

        t_windows = sliding_window_view(turnover, window_shape=window_size) # 注意: 前 w-1 个不够window_size 不会创建
        r_windows = sliding_window_view(ret, window_shape=window_size)
        top_indices = np.argsort(t_windows, axis=1)[:, -N:] # t_windows: 行为T日、列为T日对应的20个rolling窗口、-N: 取倒数第N个之后的数据、top_indices是索引
        top_ret = np.take_along_axis(r_windows, top_indices, axis=1)
        factor_values = np.mean(top_ret, axis=1)

        pad = np.full(window_size - 1, np.nan) # 补齐前window_size - 1 数据不足而产生的 NaN
        full_factor = np.concatenate([pad, factor_values])
        return pd.Series(full_factor, index=group.index)
    
    log("======= 计算因子 M_high =======")
    m_high = df_return.groupby('order_book_id', group_keys=False).apply(calc_rolling_top_n_sum) # group_keys=False 禁止 Pandas 在结果索引中自动添加分组键 (order_book_id) 否则变成三层索引 [obi, obi, date]
    m_high = pd.DataFrame(m_high.loc[(m_high.index.get_level_values('date') >= start_date) & (m_high.index.get_level_values('date') <= end_date)])
    m_high.columns = ['M_high']
    return m_high


if __name__ == '__main__':
    # high_5 = M_high_2_0('2016-01-01', '2025-12-31', N=5)
    # high_10 = M_high_2_0('2016-01-01', '2025-12-31', N=10)
    # high_15 = M_high_2_0('2016-01-01', '2025-12-31', N=15)
    # high_20 = M_high_2_0('2016-01-01', '2025-12-31', N=20)
    # high_5.to_parquet('high_5.parquet')
    # high_10.to_parquet('high_10.parquet')
    # high_15.to_parquet('high_15.parquet')
    # high_20.to_parquet('high_20.parquet')
    period = 1
    df_factor = pd.read_parquet('high_5.parquet')
    run_factor_test(df_factor, '2016-01-01', '2025-12-31', factor_col='M_high', bins=10, N=period)
    df_factor = pd.read_parquet('high_10.parquet')
    run_factor_test(df_factor, '2016-01-01', '2025-12-31', factor_col='M_high', bins=10, N=period)
    df_factor = pd.read_parquet('high_15.parquet')
    run_factor_test(df_factor, '2016-01-01', '2025-12-31', factor_col='M_high', bins=10, N=period)
    df_factor = pd.read_parquet('high_20.parquet')
    run_factor_test(df_factor, '2016-01-01', '2025-12-31', factor_col='M_high', bins=10, N=period)
    print(1)