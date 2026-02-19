---
inclusion: always
---

# 技术栈

## 核心库
- pandas: 数据处理和时间序列分析
- numpy: 数值计算
- matplotlib: 可视化和绘图
- Python 3.x 配合虚拟环境 (.venv)

## 数据格式
- 主要数据存储: Parquet 文件（通过 .gitignore 排除在 git 之外）
- 数据组织: 按年份分区以实现高效加载
- 关键数据集: K线数据（OHLC）、因子值、交易状态标志

## 项目结构
- `factor_establish/`: 因子计算实现
- `factor_backtest_system/`: 回测引擎和工具
- 根目录: 数据文件（.parquet）和分析笔记本

## 常用命令

### 环境设置
```bash
# 激活虚拟环境
source .venv/bin/activate  # Linux/Mac
.venv\Scripts\activate     # Windows
```

### 运行因子计算
```python
# 示例: 计算 STR 因子
from factor_establish.STR_demo import STR_FACTOR
df = STR_FACTOR('2020-01-01', '2023-12-31', sita=0.1, delta=0.7, window_size=21)
```

### 运行回测
```python
# 示例: 运行因子回测
from factor_backtest_system.factor_engine_fast import run_factor_test
results = run_factor_test(df_factor, start_date, end_date, factor_col='str_factor', bins=5, N=1)
```

## 代码风格约定
- 中文注释和文档字符串是可接受的（项目处于中文环境）
- 函数命名: snake_case
- 适当使用类型提示
- 通过 common_utils.py 中的自定义 `log()` 函数记录日志
