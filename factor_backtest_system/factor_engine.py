import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from backtest_utils import *
from common_utils import *

# 1. 描述性统计、覆盖度分析
def plot_factor_stats(df_factor):
    """
    图1：因子统计特征
    Subplot 1: 全历史频率分布图 (Histogram + KDE)
    Subplot 2: 时序均值与标准差通道
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 12))
    
    # --- 子图 1: 全历史频率分布 ---
    # 使用 seaborn 的 displot 绘制密度曲线，能直观看出因子是否符合正态分布
    sns.histplot(df_factor['factor_value'], kde=True, ax=axes[0], color='#2ca02c', stat='density')
    axes[0].set_title('Global Factor Value Distribution (Density)')
    axes[0].set_xlabel('Factor Value')
    axes[0].grid(True, linestyle='--', alpha=0.5)

    # --- 子图 2: 时序均值与标准差通道 ---
    daily_stats = df_factor.groupby('date')['factor_value'].agg(['mean', 'std'])
    axes[1].plot(daily_stats.index, daily_stats['mean'], label='Daily Mean', color='#1f77b4')
    axes[1].fill_between(daily_stats.index, 
                         daily_stats['mean'] - daily_stats['std'], 
                         daily_stats['mean'] + daily_stats['std'], 
                         color='#1f77b4', alpha=0.2, label='1 Std Dev Channel')
    axes[1].set_title('Factor Value Time Series (Mean & Std Dev Channel)')
    axes[1].set_ylabel('Factor Value')
    axes[1].legend(loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def plot_factor_coverage(df_factor, df_daily):
    """
    图2：因子覆盖度时序图
    """
    # 统计每日因子样本数 vs 全A行情样本数
    factor_count = df_factor.groupby('date').size()
    all_a_count = df_daily.groupby('date').size()
    coverage = factor_count / all_a_count

    plt.figure(figsize=(12, 5))
    plt.plot(coverage.index, coverage, color='#d62728', label='Full-A Coverage Ratio')
    plt.fill_between(coverage.index, 0, coverage, color='#d62728', alpha=0.1)
    
    plt.title('Factor Coverage Ratio Over Time')
    plt.ylabel('Coverage Ratio (%)')
    plt.ylim(0, 1.1) # 比例通常在 0-1 之间
    plt.legend(loc='upper left')
    plt.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plt.show()

def get_future_returns(df_close_wide, p=1):
    """计算未来收益率 (矩阵化)"""
    return (df_close_wide.shift(-p) / df_close_wide) - 1
 
def preprocess_factor(wide_factor):
    """因子预处理：去极值 (Winsorize) 和 标准化 (Z-Score)"""
    # 截面去极值 (3倍中位数绝对偏差法)
    def winsorize_series(s):
        median = s.median()
        mad = (s - median).abs().median()
        threshold = 3 * 1.4826 * mad
        return s.clip(lower=median - threshold, upper=median + threshold)
    
    # 截面处理
    wide_factor = wide_factor.apply(winsorize_series, axis=1)
    # 截面标准化
    wide_factor = wide_factor.apply(lambda s: (s - s.mean()) / s.std(), axis=1)
    return wide_factor
 
def calc_ic_stats(wide_factor, wide_ret):
    """计算 IC 和 Rank IC"""
    ic = wide_factor.corrwith(wide_ret, axis=1, method='pearson')
    rank_ic = wide_factor.corrwith(wide_ret, axis=1, method='spearman')
    return pd.DataFrame({'IC': ic, 'Rank_IC': rank_ic})
 
def calc_quantile_stats(wide_factor, wide_ret, bins=5):
    """分层收益计算"""
    wide_ranks = wide_factor.rank(axis=1, pct=True)
    wide_quantiles = (wide_ranks * bins).apply(np.ceil)
    
    res = {}
    for b in range(1, bins + 1):
        res[f'Group_{b}'] = wide_ret[wide_quantiles == b].mean(axis=1)
    return pd.DataFrame(res)

def run_factor_test(df_factor, start_date, end_date, factor_col='str_factor', periods=[1], bins=5):
    """
    一键运行全流程：长表输入 -> 宽表计算 -> 结果输出
    """
    # --- 1. 数据准备 (长转宽) ---
    log("======= 数据准备 =======")
    start_year = pd.to_datetime(start_date).year
    end_year = pd.to_datetime(end_date).year
    df_daily = load_parquet_by_years('daily', 'daily', start_year, end_year)
    df_daily = df_daily.loc[pd.IndexSlice[:, start_date:end_date], :]

    wide_factor = df_factor[factor_col].unstack(level='order_book_id')
    wide_close = df_daily['close'].unstack(level='order_book_id')
    
    # 对齐索引
    common_dates = wide_factor.index.intersection(wide_close.index)
    wide_factor = wide_factor.loc[common_dates]
    wide_close = wide_close.loc[common_dates]

    # --- 2. 预处理 ---
    # 先处理因子值
    log("======= 因子预处理 =======")
    processed_factor = preprocess_factor(wide_factor)

    # --- 3. 计算收益率 ---
    # 默认分析 next_1_ret
    log("======= 收益率计算 =======")
    wide_ret = get_future_returns(wide_close, p=periods[0])

    # --- 4. 评价 ---
    log("======= IC分析 =======")
    ic_df = calc_ic_stats(processed_factor, wide_ret)

    log("======= 分层汇总 =======")
    quantile_df = calc_quantile_stats(processed_factor, wide_ret, bins=bins)
    
    # --- 5. 汇总结果 ---
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
    
    return results

if __name__ == '__main__':
    df_factor = pd.read_parquet(r'G:\quant_road\str.parquet')
    run_factor_test(df_factor, '2016-01-01', '2025-12-31')
    print(1)