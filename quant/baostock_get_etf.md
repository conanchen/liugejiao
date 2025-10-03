# Baostock ETF 数据获取方法（更新版）

## 重要提醒：经测试，当前版本的 Baostock **不支持** 获取 ETF 数据

通过实际测试发现，Baostock 目前无法获取 ETF 相关数据，但可以正常获取指数数据。以下是测试结果和替代方案：

## 测试结果总结

### 1. ETF 数据获取尝试

#### 1.1 获取ETF代码列表
使用`query_all_stock`方法尝试获取ETF代码列表，测试了两种参数设置，并将结果保存到CSV文件中：

- **参数设置1**：添加日期参数（`day="2024-10-25"`）
  - API返回状态：成功（错误码=0，错误信息=success）
  - 实际结果：成功遍历了5646个代码，但仍然未能找到任何以510/159开头的ETF代码
  - 结果文件：`all_stock_with_day.csv`

- **参数设置2**：移除日期参数（仅调用`query_all_stock()`）
  - API返回状态：成功（错误码=0，错误信息=success）
  - 实际结果：成功遍历了5657个代码，但仍然未能找到任何以510/159开头的ETF代码
  - 结果文件：`all_stock_without_day.csv`

#### 1.2 获取ETF K线数据
尝试直接获取常见ETF的K线数据：
- **测试代码**：sh.510300（沪深300ETF）、sz.159915（创业板ETF）
- **时间范围**：2023-01-01 至 2023-12-31
- **结果**：均未获取到任何K线数据

### 2. 指数数据获取（可用）
- **测试代码**：sh.000300（沪深300指数）、sz.399006（创业板指数）
- **时间范围**：2023-01-01 至 2023-12-31
- **结果**：均成功获取到完整的K线数据
- **数据包含**：日期、开盘价、最高价、最低价、收盘价、成交量等字段
- **数据样例**：
  ```
        date      code      open      high       low     close      volume        
  2023-01-03 sh.000300 3864.8356 3893.9904 3831.2450 3887.8992 11505187500        
  2023-01-04 sh.000300 3886.2504 3905.9032 3873.6549 3892.9477 11066074300        
  2023-01-05 sh.000300 3913.4928 3974.8750 3912.2600 3968.5782 11650239500        
  ```

## 替代方案：使用指数数据

如果需要进行类似 ETF 的策略回测，建议使用对应的指数数据作为替代：

```python
import baostock as bs
import pandas as pd

# 1. 登录 Baostock
lg = bs.login()
if lg.error_code == '0':
    print("登录成功")
else:
    print(f"登录失败: {lg.error_msg}")

# 2. 获取指数数据（替代 ETF 数据）
# 使用指数代码替代 ETF 代码
index_code = "sh.000300"  # 沪深300指数，替代 sh.510300（沪深300ETF）

rs = bs.query_history_k_data_plus(
    index_code,
    "date,code,open,high,low,close,volume",
    start_date="2023-01-01",
    end_date="2023-12-31",
    frequency="d",
    adjustflag="3"  # 前复权
)

# 处理数据
if rs.error_code == '0':
    data_list = []
    while rs.next():
        data_list.append(rs.get_row_data())
    df = pd.DataFrame(data_list, columns=rs.fields)
    print(f"成功获取 {index_code} 的数据")
    print(df.tail())
else:
    print(f"获取数据失败: {rs.error_msg}")

# 3. 登出
bs.logout()
```

## 建议

1. **使用指数数据替代**：由于 Baostock 不支持 ETF 数据，建议使用对应的指数数据进行策略开发和回测

2. **添加错误处理**：在实际应用中，添加完善的异常处理逻辑，以应对网络问题或数据不可用的情况

3. **考虑其他数据源**：如果确实需要 ETF 数据，可以考虑使用其他金融数据 API 或数据源

4. **关注 API 更新**：Baostock 可能会在未来版本中增加 ETF 数据支持，请关注其官方更新

## 测试脚本

### 1. test_baostock_etf_methods.py
包含了完整的ETF和指数数据获取测试流程，包括登录验证、query_all_stock方法测试、特定ETF代码和指数代码的K线数据获取测试，以及详细的结果验证。

### 2. save_all_stock_data.py
专门用于将query_all_stock()的结果保存到CSV文件的脚本，测试了两种参数设置：
- 带日期参数（`day="2024-10-25"`）
- 不带日期参数

脚本会将两种方式获取的结果分别保存为：
- `all_stock_with_day.csv`（5646个代码）
- `all_stock_without_day.csv`（5657个代码）

同时会自动检查结果中是否存在以510/159开头的ETF代码。