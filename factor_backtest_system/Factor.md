# 单因子回测框架 — 时序规则

## 核心时序假设

| 时间 | 动作 | 约束 |
|------|------|------|
| T 日收盘 | 因子值确定（截面排名、分组） | — |
| T+1 日 | 按分组以 close 价买入 | T+1 日 `can_buy == True` |
| T+2 日 | 以 close 价卖出，实现收益 | T+2 日 `can_sell == True` |

## 收益率定义

```
future_return(T) = close(T+2) / close(T+1) - 1
```

即：T 日这一行记录的收益率，是 T+1 日买入、T+2 日卖出的收益。

## shift(-1) 对齐逻辑

代码中的两行 shift：

```python
wide_can_buy_aligned  = wide_can_buy.shift(-1)   # T日行 ← T+1日的 can_buy
wide_can_sell_aligned = wide_can_sell.shift(-1)   # T日行 ← T+1日的 can_sell
```

原因：`can_buy` / `can_sell` 原始数据每行记录的是**当天**的状态，但我们站在 **T 日**做决策，实际执行发生在 **T+1 日**。`shift(-1)` 把 T+1 日的值"拉"到 T 日这一行，使决策日和执行日对齐。

示例（`can_sell`）：

| 日期 | can_sell（原始） | can_sell.shift(-1) |
|------|------------------|--------------------|
| 1月2日 | True | **False**（1月3日的值） |
| 1月3日 | False（跌停） | True（1月4日的值） |
| 1月4日 | True | ... |

→ 站在 T=1月2日，检查 `shift(-1)` 后的值 = False，说明 T+1=1月3日 卖不出 → 被迫继续持有。

这**不是未来函数**：回测框架模拟的是"T+1日执行时才发现能不能交易"这个现实，并非在 T 日偷看未来做决策。

## 交易约束处理（精细版，逐日模拟持仓）

### 买入端
- **因子 mask**：`buyable_factor = wide_factor.where(wide_can_buy.shift(-1))`
  - T+1 日不可买入的股票，T 日因子设 NaN，不参与分组
  - 只有 T+1 日 `can_buy == True` 的股票才会被纳入目标持仓

### 卖出端（无未来函数）
- **不直接 mask 收益**，而是逐日模拟持仓：
  - 每个交易日，检查上期持仓中哪些股票 T+1 日 `can_sell == False`
  - 卖不出的股票**被迫继续持有**，其收益仍计入组合
  - 实际持仓 = 当期目标持仓（新买入） ∪ 上期卖不出的（强制持有）
  - 组合收益 = 实际持仓内所有股票的等权平均收益

## can_buy / can_sell 定义（见 backtest_utils.py）

- `can_buy = (非ST) & (非停牌) & (非涨停)`
- `can_sell = (非ST) & (非停牌) & (非跌停)`
