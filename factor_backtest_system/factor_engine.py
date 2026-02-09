import seaborn as sns
import matplotlib.pyplot as plt

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

d