import os
import pandas as pd
import numpy as np
from datetime import datetime
from common_utils import log

DATA_ROOT = r'D:\Data_Root'

def load_parquet_by_years(folder, prefix, start_year, end_year):
    """
    加载指定文件夹内跨年分 Parquet 文件并合并
    """
    dfs = []
    for year in range(start_year, end_year + 1):
        file_path = os.path.join(DATA_ROOT, folder, f'{prefix}_{year}.parquet')
        if os.path.exists(file_path):
            df = pd.read_parquet(file_path)
            dfs.append(df)
    
    if not dfs:
        return pd.DataFrame()
    
    return pd.concat(dfs).sort_index()

def get_clean_factor_data(start_date, end_date):
    """
    获取基础对齐数据：行情、市值、交易状态
    返回 MultiIndex (order_book_id, date)
    """
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year

    log("======= 加载日线行情数据 =======")
    df_daily = load_parquet_by_years('daily', 'daily', start_year, end_year)

    log("======= 加载交易约束(ST, 停牌, 涨跌停) =======")
    df_limit = load_parquet_by_years('trading_details', 'limits', start_year, end_year)
    df_suspended = load_parquet_by_years('trading_details', 'suspended', start_year, end_year)
    df_st = load_parquet_by_years('trading_details', 'st', start_year, end_year)

    df_daily = df_daily.join(df_limit)

    df_suspended_long = df_suspended.T.stack().to_frame('is_suspended')
    df_suspended_long.index.names = ['order_book_id', 'date']
    df_daily = df_daily.join(df_suspended_long)
    df_daily['is_suspended'] = df_daily['is_suspended'].fillna(False)

    df_st_long = df_st.T.stack().to_frame('is_st')
    df_st_long.index.names = ['order_book_id', 'date']
    df_daily = df_daily.join(df_st_long)
    df_daily['is_st'] = df_daily['is_st'].fillna(False)

    df_daily = df_daily.loc[pd.IndexSlice[:, start_date:end_date], :]

    log("======= 数据加载完成 =======")
    return df_daily

def add_trading_status(df):
    """
    过滤: 剔除ST、停牌、涨跌停无法交易的股票
    """

    log("======= 生成交易状态标签(can buy/ can sell) =======")
    # 当天是否可以交易(买\卖)
    is_limit_up = (df['close'] >= df['limit_up']) & (df['limit_up'] > 0)
    is_limit_down = (df['close'] <= df['limit_down']) & (df['limit_down'] > 0)
    
    df['can_buy'] = (df['is_st'] == False)  & \
                    (df['is_suspended'] == False) & \
                    (~is_limit_up)
    
    df['can_sell'] = (df['is_st'] == False) & \
                     (df['is_suspended'] == False) & \
                     (~is_limit_down)
    
    return df

def get_K_data(start_date, end_date):
    df_daily = get_clean_factor_data(start_date, end_date)
    df_clean = add_trading_status(df_daily)
    return df_clean

if __name__ == '__main__':
    df = get_K_data('2015-01-01', '2015-12-31')