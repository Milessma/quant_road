import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from backtest_utils import *
from common_utils import *
import tqdm

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

def plot_ic_cumulative(ic_df):
    """
    图3：累积 IC / Rank IC 时序图
    上图: IC 柱状图 + 累积IC曲线
    下图: Rank IC 柱状图 + 累积Rank IC曲线
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # --- 上图: IC ---
    ic = ic_df['IC']
    cum_ic = ic.cumsum()
    ic_mean = ic.mean()
    ic_std = ic.std()
    ic_ir = ic_mean / ic_std if ic_std != 0 else 0
    ic_pos_ratio = (ic > 0).sum() / len(ic)

    colors_ic = np.where(ic >= 0, '#e74c3c', '#2ecc71')
    axes[0].bar(ic.index, ic, color=colors_ic, alpha=0.6, width=1.0)
    ax0_twin = axes[0].twinx()
    ax0_twin.plot(cum_ic.index, cum_ic, color='#2c3e50', linewidth=1.5, label='Cumulative IC')
    ax0_twin.set_ylabel('Cumulative IC')
    ax0_twin.legend(loc='upper left')

    axes[0].set_ylabel('IC')
    axes[0].set_title(
        f'IC Series  |  Mean={ic_mean:.4f}  Std={ic_std:.4f}  '
        f'IR={ic_ir:.4f}  IC>0: {ic_pos_ratio:.1%}'
    )
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # --- 下图: Rank IC ---
    rank_ic = ic_df['Rank_IC']
    cum_rank_ic = rank_ic.cumsum()
    ric_mean = rank_ic.mean()
    ric_std = rank_ic.std()
    ric_ir = ric_mean / ric_std if ric_std != 0 else 0
    ric_pos_ratio = (rank_ic > 0).sum() / len(rank_ic)

    colors_ric = np.where(rank_ic >= 0, '#e74c3c', '#2ecc71')
    axes[1].bar(rank_ic.index, rank_ic, color=colors_ric, alpha=0.6, width=1.0)
    ax1_twin = axes[1].twinx()
    ax1_twin.plot(cum_rank_ic.index, cum_rank_ic, color='#2c3e50', linewidth=1.5, label='Cumulative Rank IC')
    ax1_twin.set_ylabel('Cumulative Rank IC')
    ax1_twin.legend(loc='upper left')

    axes[1].set_ylabel('Rank IC')
    axes[1].set_title(
        f'Rank IC Series  |  Mean={ric_mean:.4f}  Std={ric_std:.4f}  '
        f'IR={ric_ir:.4f}  RankIC>0: {ric_pos_ratio:.1%}'
    )
    axes[1].axhline(y=0, color='black', linewidth=0.5)
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def plot_quantile_returns(quantile_df, bins=5):
    """
    图4：分层收益图
    子图1: 各组平均日收益柱状图
    子图2: 各组累积净值曲线
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # 配色：从深绿到深红，避免明黄色
    colors = plt.cm.RdYlGn_r(np.linspace(0.1, 0.9, bins))

    # --- 子图1: 各组平均日收益柱状图 ---
    mean_rets = quantile_df.mean()
    axes[0].bar(mean_rets.index, mean_rets.values, color=colors, edgecolor='grey', linewidth=0.5)
    for i, (name, val) in enumerate(mean_rets.items()):
        axes[0].text(i, val, f'{val*100:.4f}%', ha='center',
                     va='bottom' if val >= 0 else 'top', fontsize=9)
    axes[0].set_title('Average Daily Return by Quantile Group')
    axes[0].set_ylabel('Mean Daily Return')
    axes[0].axhline(y=0, color='black', linewidth=0.5)
    axes[0].grid(True, linestyle='--', alpha=0.3, axis='y')

    # --- 子图2: 各组累积净值曲线 ---
    for i, col in enumerate(quantile_df.columns):
        cum_ret = (1 + quantile_df[col]).cumprod()
        axes[1].plot(cum_ret.index, cum_ret, label=col, color=colors[i], linewidth=1.2)

    axes[1].set_title('Quantile Cumulative Net Value')
    axes[1].set_ylabel('Net Value')
    axes[1].legend(loc='upper left', ncol=bins)
    axes[1].axhline(y=1, color='black', linewidth=0.5)
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def _calc_drawdown(nav):
    """计算回撤序列：返回负值序列（从高点回落的比例）"""
    running_max = nav.cummax()
    drawdown = (nav - running_max) / running_max
    return drawdown

def plot_long_short(quantile_df, bins=5):
    """
    图5：多头/多空净值与回撤
    子图1: 多头(Group_bins)净值曲线 + 回撤面积图
    子图2: 多空(Group_bins - Group_1)净值曲线 + 回撤面积图
    """
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # ========== 子图1: 多头 ==========
    long_ret = quantile_df[f'Group_{bins}']
    long_nav = (1 + long_ret).cumprod()
    long_dd = _calc_drawdown(long_nav)
    long_ann = long_ret.mean() * 242
    long_max_dd = long_dd.min()

    axes[0].plot(long_nav.index, long_nav, color='#e74c3c', linewidth=1.2, label='Long NAV')
    ax0_twin = axes[0].twinx()
    ax0_twin.fill_between(long_dd.index, 0, long_dd, color='#bdc3c7', alpha=0.5, label='Drawdown')
    ax0_twin.set_ylabel('Drawdown')
    ax0_twin.set_ylim(long_dd.min() * 1.5, 0)

    axes[0].set_ylabel('Net Value')
    axes[0].set_title(
        f'Long (Group_{bins})  |  '
        f'Annualized≈{long_ann:.2%}  MaxDD={long_max_dd:.2%}'
    )
    h1, l1 = axes[0].get_legend_handles_labels()
    h2, l2 = ax0_twin.get_legend_handles_labels()
    axes[0].legend(h1 + h2, l1 + l2, loc='upper left')
    axes[0].grid(True, linestyle='--', alpha=0.3)

    # ========== 子图2: 多空 ==========
    ls_ret = quantile_df[f'Group_{bins}'] - quantile_df['Group_1']
    ls_nav = (1 + ls_ret).cumprod()
    ls_dd = _calc_drawdown(ls_nav)
    ls_ann = ls_ret.mean() * 242
    ls_max_dd = ls_dd.min()

    axes[1].plot(ls_nav.index, ls_nav, color='#2c3e50', linewidth=1.2, label='L/S NAV')
    ax1_twin = axes[1].twinx()
    ax1_twin.fill_between(ls_dd.index, 0, ls_dd, color='#bdc3c7', alpha=0.5, label='Drawdown')
    ax1_twin.set_ylabel('Drawdown')
    ax1_twin.set_ylim(ls_dd.min() * 1.5, 0)

    axes[1].set_ylabel('Net Value')
    axes[1].set_title(
        f'Long-Short (Group_{bins} - Group_1)  |  '
        f'Annualized≈{ls_ann:.2%}  MaxDD={ls_max_dd:.2%}'
    )
    h1, l1 = axes[1].get_legend_handles_labels()
    h2, l2 = ax1_twin.get_legend_handles_labels()
    axes[1].legend(h1 + h2, l1 + l2, loc='upper left')
    axes[1].grid(True, linestyle='--', alpha=0.3)

    plt.tight_layout()
    plt.show()

def get_future_returns(df_close_wide):
    """
    计算未来收益率 (矩阵化)
    T日行对应的收益 = close(T+2) / close(T+1) - 1
    即: T日看因子, T+1日买入, T+2日卖出
    """
    return df_close_wide.shift(-2) / df_close_wide.shift(-1) - 1
 
def preprocess_factor(wide_factor, n=3):
    """因子预处理：去极值 (Winsorize) 和 标准化 (Z-Score)"""
    # 截面去极值 (3倍中位数绝对偏差法)
    def winsorize_series(s, q=n):
        median = s.median()
        mad = (s - median).abs().median()
        threshold = q * 1.4826 * mad
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
 
def calc_quantile_stats(wide_factor, wide_ret, wide_can_buy, wide_can_sell, bins=5):
    """
    分层收益计算 (精细版: 考虑交易约束)
    
    逻辑:
    - T日因子分组 → T+1日买入 → T+2日卖出
    - wide_can_buy: T+1日能否买入 (已 shift(-1) 对齐到T日行)
    - wide_can_sell: T+1日能否卖出上期持仓 (已 shift(-1) 对齐到T日行)
    - 卖不出的股票被迫继续持有, 其收益仍计入组合
    """
    # 1. 只对可买入的股票进行分组
    buyable_factor = wide_factor.where(wide_can_buy)
    wide_ranks = buyable_factor.rank(axis=1, pct=True)
    wide_quantiles = (wide_ranks * bins).apply(np.ceil)
    
    dates = wide_factor.index
    prev_holdings = {b: set() for b in range(1, bins + 1)}
    res = {f'Group_{b}': [] for b in range(1, bins + 1)}
    
    for i, date in tqdm.tqdm(enumerate(dates), total=len(dates), desc='分层持仓模拟'):
        for b in range(1, bins + 1):
            # 当期目标持仓: 因子分组中属于 group b 的股票
            target_stocks = set(wide_quantiles.columns[wide_quantiles.loc[date] == b])
            
            # 上期持仓中, 今天卖不出的股票 (can_sell 为 False)
            forced_hold = set()
            for stk in prev_holdings[b]:
                if stk in wide_can_sell.columns and pd.notna(wide_can_sell.loc[date, stk]):
                    if not wide_can_sell.loc[date, stk]:
                        forced_hold.add(stk)
            
            # 实际持仓 = 目标持仓中可买入的 + 上期卖不出的
            actual_holdings = target_stocks | forced_hold
            
            # 计算组合等权收益
            if len(actual_holdings) > 0:
                rets = wide_ret.loc[date, list(actual_holdings)].dropna()
                group_ret = rets.mean() if len(rets) > 0 else 0.0
            else:
                group_ret = 0.0
            
            res[f'Group_{b}'].append(group_ret)
            prev_holdings[b] = actual_holdings
    
    return pd.DataFrame(res, index=dates)

def run_factor_test(df_factor, start_date, end_date, factor_col='str_factor', bins=5):
    """
    一键运行全流程：长表输入 -> 宽表计算 -> 结果输出
    时序: T日因子 → T+1日买入 → T+2日卖出
    """
    # --- 1. 数据准备 (长转宽) ---
    log("======= 数据准备(含交易约束) =======")
    df_daily = get_K_data(start_date, end_date)

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

    # shift 对齐到 T 日视角:
    # T+1日能否买入 → shift(-1) 使其对齐到 T 日行
    # T+1日能否卖出上期持仓 → shift(-1) 使其对齐到 T 日行
    wide_can_buy_aligned = wide_can_buy.shift(-1)
    wide_can_sell_aligned = wide_can_sell.shift(-1)

    # --- 2. 预处理 ---
    log("======= 因子预处理 =======")
    processed_factor = preprocess_factor(wide_factor)

    # --- 3. 计算收益率 ---
    log("======= 收益率计算 =======")
    wide_ret = get_future_returns(wide_close)

    # --- 4. 评价 ---
    log("======= IC分析 =======")
    ic_df = calc_ic_stats(processed_factor, wide_ret)

    log("======= 分层汇总(含交易约束) =======")
    quantile_df = calc_quantile_stats(
        processed_factor, wide_ret, wide_can_buy_aligned, wide_can_sell_aligned, bins=bins
    )
    
    # --- 5. 可视化 ---
    log("======= 绘制IC累积图 =======")
    plot_ic_cumulative(ic_df)
    log("======= 绘制分层收益图 =======")
    plot_quantile_returns(quantile_df, bins=bins)
    log("======= 绘制多头/多空净值与回撤 =======")
    plot_long_short(quantile_df, bins=bins)

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
    df_factor = pd.read_parquet(r'G:\quant_road\str.parquet')
    run_factor_test(df_factor, '2016-01-01', '2025-12-31', bins=10)
    print(1)