---
inclusion: always
---

# 项目结构

## 目录组织

```
/
├── factor_establish/          # 因子计算模块
│   ├── STR_demo.py           # 显著性理论收益率因子
│   ├── W_reverse.py          # 反转因子（M_high, M_low, W_reverse）
│   ├── W_reverse_change.py   # 修改后的反转因子（M_high_2_0）
│   ├── Factor.md             # 因子理论文档
│   └── image/                # 文档图片
│
├── factor_backtest_system/    # 回测框架
│   ├── backtest_utils.py     # 数据加载和预处理工具
│   ├── config.py             # 交易成本参数（佣金、印花税、滑点）
│   ├── factor_engine.py      # 可视化函数（IC、分位数收益、覆盖率）
│   ├── factor_engine_fast.py # 核心回测引擎（含交易约束）
│   └── engine_q.md           # 回测时间序列逻辑文档
│
├── common_utils.py            # 共享工具（日志函数）
├── *.parquet                  # 数据文件（已忽略 git）
├── view.ipynb                 # 分析笔记本
└── question.md                # 开发笔记和优化记录
```

## 关键架构模式

### 数据流
1. 通过 backtest_utils 中的 `load_parquet_by_years()` 加载原始数据
2. 通过 `add_trading_status()` 添加交易状态标志（can_buy, can_sell）
3. 在 factor_establish 模块中计算因子值
4. 通过 `run_factor_test()` 运行回测，确保正确的时间对齐
5. 使用 factor_engine 绘图函数可视化结果

### 时间对齐逻辑
- 因子数据使用 `shift(-1)` 将 T 日决策与 T+1 执行对齐
- 在执行时（而非决策时）检查交易约束，避免未来函数
- 收益率计算为: `close(T+2) / close(T+1) - 1`

### 交易约束
- `can_buy`: 排除 ST 股票、停牌股票、涨停股票
- `can_sell`: 排除 ST 股票、停牌股票、跌停股票
- 无法卖出的持仓被强制持有，影响组合构成

### 配置
- 交易成本集中在 `config.py` 中
- 印花税率变化按日期处理（2023-08-28 阈值）
- 默认参数: 佣金 0.01%、滑点 0.1%、印花税 0.05%/0.1%
