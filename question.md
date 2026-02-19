# 回测系统优化记录

## 1. 数据加载年份区间修正
在 `load_parquet_by_years` 函数中，原先使用的 `range(start_year, end_year)` 会漏掉结束年份的数据。已修正为 `range(start_year, end_year + 1)`。

## 2. 交易过滤逻辑优化（信号与交易分离）
针对“T日计算因子，T+1日调仓”的单因子回测需求，我们讨论并确定了以下设计原则：
- **避免前瞻性偏差**：不再直接在 T 日剔除涨跌停股票，以免丢失重要的因子样本。
- **状态标记法**：将 `filter_universe` 重构为生成 `can_buy` 和 `can_sell` 标签。
    - `can_buy`: 剔除 ST、停牌、封死涨停（T+1日买入受限）。
    - `can_sell`: 剔除 ST、停牌、封死跌停（T+1日卖出受限）。
- **执行逻辑**：因子计算函数保持纯净，具体的交易约束（如 T+1 能否买入）交给专门的调仓函数处理。

## 3. 待办事项
- [x] 修复 `range` bug。
- [ ] 完成 `filter_universe` 向状态标记函数的重构。
- [ ] 实现基于 T+1 交易标签的调仓逻辑。

## 4. 收益率计算方式修正（累计收益率 vs 日收益率）

### 问题背景
原先的 `calc_quantile_stats_fast` 函数在计算组合收益时，使用的是每日收益率的简单平均：
```python
r = ret_arr[t]  # 当日的日收益率
group_rets[t, b] = r[mask].sum() / cnt
```

这种方式存在问题：在持仓期内，无法正确反映相对于买入价的累计收益。

**举例说明**：
- 第 1 天以 10 元买入股票 A
- 第 2 天涨到 11 元（日收益率 10%，累计收益率 10%）
- 第 3 天涨到 12 元（日收益率 9.09%，累计收益率 20%）

原代码在第 3 天会用 9.09%，但正确的应该用 20%（相对于买入价 10 元）。

### 修正方案
改为基于买入价的累计收益率计算：

1. **记录买入基准价格**：在调仓日记录每只股票的买入价
   ```python
   if is_rebalance:
       price_base[b] = np.where(target, close_arr[t], price_base[b])
   ```

2. **计算累计收益率**：每天计算相对于买入价的累计收益
   ```python
   cum_ret = (close_arr[t] - price_base[b]) / price_base[b]
   ```

3. **等权平均**：对所有持仓股票的累计收益率取平均
   ```python
   group_rets[t, b] = cum_ret[mask].sum() / cnt if cnt > 0 else 0.0
   ```

### 关键技术点

#### `np.where` 的使用
```python
price_base[b] = np.where(target, close_arr[t], price_base[b])
```
- **语法**：`np.where(条件, if True取这个值, if False取这个值)`
- **作用**：对数组的每个元素进行条件判断
- **本例中**：
  - 如果是新买入的股票（`target=True`）：记录当前价格 `close_arr[t]`
  - 如果不是新买入的股票（`target=False`）：保留原价格 `price_base[b]`
- **为什么不能用 `np.nan`**：如果用 `np.where(target, close_arr[t], np.nan)`，会把强制持有股票的买入价清空

**示例**：
```python
target = [True, False, True, False]  # 4只股票，第1、3只是目标
close_arr[t] = [10, 20, 30, 40]      # 当前价格
price_base[b] = [NaN, 15, NaN, 25]   # 之前的买入价

# 错误写法：
price_base[b] = np.where(target, close_arr[t], np.nan)
# 结果：[10, NaN, 30, NaN]  # 第2、4只股票的买入价丢失了！

# 正确写法：
price_base[b] = np.where(target, close_arr[t], price_base[b])
# 结果：[10, 15, 30, 25]  # 保留了强制持有股票的买入价
```

#### 交易约束的时间对齐
- `wide_factor`：经过 `shift(1)` 后，索引 T 对应的是 T-1 日的因子值
- `wide_can_buy/wide_can_sell`：索引 T 对应的是 T 日的交易约束
- `wide_close`：索引 T 对应的是 T 日的收盘价

**时间线示例**（以索引 `2024-01-05` 为例）：

| 数据 | 含义 | 时间 |
|------|------|------|
| `wide_factor[2024-01-05]` | 2024-01-04 的因子值（已 shift） | T-1 日 |
| `wide_can_buy[2024-01-05]` | 2024-01-05 能否买入 | T 日 |
| `wide_can_sell[2024-01-05]` | 2024-01-05 能否卖出 | T 日 |
| `wide_close[2024-01-05]` | 2024-01-05 的收盘价 | T 日 |

#### 调仓日的交易逻辑

**1. 买入约束**：在分组时处理（第 40 行）
```python
buyable_factor = wide_factor.where(wide_can_buy)
```
- 用 T 日的 `can_buy` 约束过滤 T-1 日的因子值
- 只对 T 日能买的股票进行分组
- **避免"买不到"的问题**：涨停、停牌的股票不会进入目标持仓

**2. 卖出约束**：在持仓更新时处理（第 82 行）
```python
forced = prev_hold[b] & ~sell_ok  # 上期持有 且 今天卖不出
actual = target | forced           # 实际持仓 = 目标持仓 + 强制持仓
```
- 卖不出的股票（跌停、停牌）被强制继续持有
- 保留原买入价，继续计算累计收益

### 修改清单
- [x] 函数签名添加 `wide_close` 参数
- [x] 添加 `close_arr = wide_close.values`
- [x] 调仓日记录买入价：`price_base[b] = np.where(target, close_arr[t], price_base[b])`
- [x] 用累计收益率替代日收益率：`cum_ret = (close_arr[t] - price_base[b]) / price_base[b]`
- [x] 调用处传入 `wide_close`：`calc_quantile_stats_fast(processed_factor, wide_close, ...)`

## 5. 强制持有股票的处理策略

### 问题场景
假设调仓周期是 5 天（`hold_days=5`）：

**第 1 天（调仓日）**：买入股票 A、B、C

**第 2 天（非调仓日）**：股票 B 跌停，卖不出，被强制持有

**第 3 天（非调仓日）**：股票 B 不再跌停，可以卖了

**问题**：第 3 天应该卖出 B 吗？还是继续持有到第 6 天（下一个调仓日）？

### 策略 A：固定持仓周期（当前实现）✅

**逻辑**：
- 只在调仓日调整持仓，非调仓日不做任何交易
- 强制持有的股票，即使后来能卖了，也继续持有到下一个调仓日

**代码实现**（第 79-83 行）：
```python
if is_rebalance:
    target = target_masks[b, t]    # 调仓日: 用新因子分组
    price_base[b] = np.where(target, close_arr[t], price_base[b])
else:
    target = prev_hold[b]          # 非调仓日: 沿用上期持仓（包括强制持有的）
```

**优点**：
- ✅ 交易频率低，成本低
- ✅ 策略简单，易于理解和实现
- ✅ 符合"定期调仓"的设计初衷
- ✅ 保持策略的一致性（只在调仓日根据因子决策）
- ✅ 符合因子回测的标准做法

**缺点**：
- ❌ 强制持有的股票即使能卖也不卖，可能错过止损机会
- ❌ 被动持有时间可能较长

**适用场景**：
- 标准的因子回测
- 追求低交易成本的策略
- 强调持仓周期一致性的策略

### 策略 B：机会卖出（备选方案）

**逻辑**：
- 非调仓日，如果强制持有的股票能卖了，立即卖出
- 只保留"原目标持仓中能卖的"+"卖不出的强制持仓"

**代码实现**（需要修改）：
```python
if is_rebalance:
    target = target_masks[b, t]
    prev_target[b] = target  # 需要额外记录调仓日的目标持仓
    price_base[b] = np.where(target, close_arr[t], price_base[b])
else:
    # 非调仓日: 只保留上期目标持仓中能卖的 + 卖不出的强制持仓
    can_hold = prev_target[b] & sell_ok  # 上期目标中能卖的，选择继续持有
    forced = prev_hold[b] & ~sell_ok     # 卖不出的，被迫持有
    target = can_hold | forced
```

**优点**：
- ✅ 更灵活，减少被动持仓时间
- ✅ 可能降低风险（及时止损）

**缺点**：
- ❌ 增加交易频率和成本
- ❌ 逻辑更复杂，需要额外记录 `prev_target`
- ❌ 可能偏离原始因子信号
- ❌ 不符合标准的"定期调仓"回测框架

**适用场景**：
- 对交易成本不敏感的策略
- 强调风险控制的策略
- 需要更灵活持仓管理的场景

### 当前选择
**采用策略 A（固定持仓周期）**，原因：
1. 符合因子回测的标准做法（定期调仓）
2. 交易成本更低
3. 逻辑更清晰，易于维护
4. 强制持有是市场约束，不是策略选择，应等到调仓日再根据因子决定

如果未来需要实现策略 B，需要：
- 添加 `prev_target` 数组记录每次调仓日的目标持仓
- 修改非调仓日的持仓更新逻辑
- 评估额外交易成本的影响
