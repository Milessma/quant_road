# Bugfix Requirements Document

## Introduction

当前的分层收益计算方式存在错误。在 `calc_quantile_stats_fast` 函数中，每天的组合收益是通过对所有持仓股票的日收益率进行简单等权平均计算的。这种方法在持仓期内会产生不准确的收益率，因为它没有正确跟踪从调仓日买入时的基准价格开始的累计收益。

正确的做法应该是：跟踪每只股票从调仓日买入时的基准价格，计算每只股票相对于买入价的累计收益率，然后对所有持仓股票的累计收益率取平均，得到组合的当日收益。

## Bug Analysis

### Current Behavior (Defect)

1.1 WHEN 在第 t 天计算分组 b 的收益时 THEN 系统使用 `group_rets[t, b] = r[mask].sum() / cnt`，其中 r 是当日的日频收益率 `ret_arr[t]`，直接对所有持仓股票的日收益率求简单平均

1.2 WHEN 股票在调仓日之后继续持有多天时 THEN 系统没有跟踪从调仓日买入时的基准价格，导致无法计算正确的累计收益率

1.3 WHEN 计算非调仓日的组合收益时 THEN 系统使用的是当日收益率的平均值，而不是基于买入价的累计收益率的平均值

### Expected Behavior (Correct)

2.1 WHEN 在调仓日 t 买入股票时 THEN 系统 SHALL 记录每只股票的买入基准价格 `price_base[b, i] = close[t, i]`

2.2 WHEN 在第 t 天计算分组 b 的收益时 THEN 系统 SHALL 计算每只持仓股票相对于其买入价的累计收益率 `(close[t, i] - price_base[b, i]) / price_base[b, i]`

2.3 WHEN 计算组合的当日收益时 THEN 系统 SHALL 对所有持仓股票的累计收益率取等权平均，得到组合的累计收益率

2.4 WHEN 股票在非调仓日继续持有时 THEN 系统 SHALL 继续使用调仓日记录的买入基准价格来计算累计收益率

### Unchanged Behavior (Regression Prevention)

3.1 WHEN 判断是否为调仓日时 THEN 系统 SHALL CONTINUE TO 使用 `(t >= 1) and ((t - 1) % hold_days == 0)` 的逻辑

3.2 WHEN 处理卖不出的股票时 THEN 系统 SHALL CONTINUE TO 将其标记为强制持有 `forced = prev_hold[b] & ~sell_ok`，并计入实际持仓 `actual = target | forced`

3.3 WHEN 确定实际持仓时 THEN 系统 SHALL CONTINUE TO 在调仓日使用新因子分组 `target_masks[b, t]`，在非调仓日沿用上期持仓 `prev_hold[b]`

3.4 WHEN 计算有效持仓时 THEN 系统 SHALL CONTINUE TO 使用 `mask = actual & np.isfinite(r)` 来过滤出有持仓且有有效数据的股票

3.5 WHEN 没有有效持仓时 THEN 系统 SHALL CONTINUE TO 返回 0.0 作为当日收益 `group_rets[t, b] = 0.0 if cnt == 0`

3.6 WHEN 更新持仓状态时 THEN 系统 SHALL CONTINUE TO 将实际持仓保存到 `prev_hold[b]` 用于下一期

3.7 WHEN 处理因子分组时 THEN 系统 SHALL CONTINUE TO 只对可买入的股票进行分组，使用 `buyable_factor = wide_factor.where(wide_can_buy)`

3.8 WHEN 返回结果时 THEN 系统 SHALL CONTINUE TO 返回 DataFrame，索引为日期，列名为 `Group_1` 到 `Group_{bins}`
