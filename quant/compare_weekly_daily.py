# @brief: 对比周线数据与日线数据

import pandas as pd
from qlib_data_fetcher import QlibDataFetcher, get_date_range
from datetime import datetime, timedelta

# 创建qlib数据获取器
fetcher = QlibDataFetcher()

# 初始化qlib
if not fetcher.initialize():
    print("初始化qlib失败")
    exit()

# 获取日期范围
start_date, end_date = get_date_range()
print(f"日期范围: {start_date} 至 {end_date}")

# 选择股票代码
stock_code = 'SH600519'  # 贵州茅台
print(f"对比股票: {stock_code}")

# 获取日线数据
print("获取日线数据...")
daily_df = fetcher.get_daily_data(stock_code, start_date, end_date)
if daily_df is None or daily_df.empty:
    print(f"获取{stock_code}的日线数据失败")
    fetcher.close()
    exit()

# 获取周线数据
print("获取周线数据...")
weekly_df = fetcher.get_weekly_data(stock_code, start_date, end_date)
if weekly_df is None or weekly_df.empty:
    print(f"获取{stock_code}的周线数据失败")
    fetcher.close()
    exit()

print(f"成功获取数据，日线数据行数: {len(daily_df)}，周线数据行数: {len(weekly_df)}")

# 提取第一周和最后一周的周线数据
first_week = weekly_df.iloc[0]
last_week = weekly_df.iloc[-1]

print("\n=== 第一周周线数据 ===")
print(f"日期: {first_week['date']}")
print(f"股票代码: {first_week['code']}")
# 获取第一周第一天的复权因子
def get_first_day_factor(df, week_date):
    week_monday = week_date - timedelta(days=week_date.weekday())
    week_days = df[(pd.to_datetime(df['date']) >= week_monday) & (pd.to_datetime(df['date']) <= week_date)]
    if not week_days.empty:
        return week_days['factor'].iloc[0]
    return 1.0

# 计算原始开盘价（除权价）
first_week_date = pd.to_datetime(first_week['date'])
first_week_first_day_factor = get_first_day_factor(daily_df, first_week_date)
first_week_original_open = first_week['open'] / first_week_first_day_factor

print(f"开盘价: {first_week['open']}")
print(f"开盘价(原始价/除权价): {first_week_original_open}")
print(f"最高价: {first_week['high']}")
print(f"最低价: {first_week['low']}")
print(f"收盘价: {first_week['close']}")
print(f"成交量: {first_week['volume']}")
print(f"成交额: {first_week['amount']}")

print("\n=== 最后一周周线数据 ===")
print(f"日期: {last_week['date']}")
print(f"股票代码: {last_week['code']}")
# 计算最后一周的原始开盘价
last_week_date = pd.to_datetime(last_week['date'])
last_week_first_day_factor = get_first_day_factor(daily_df, last_week_date)
last_week_original_open = last_week['open'] / last_week_first_day_factor

print(f"开盘价: {last_week['open']}")
print(f"开盘价(原始价/除权价): {last_week_original_open}")
print(f"最高价: {last_week['high']}")
print(f"最低价: {last_week['low']}")
print(f"收盘价: {last_week['close']}")
print(f"成交量: {last_week['volume']}")
print(f"成交额: {last_week['amount']}")

# 查找第一周对应的日线数据
# 周线是按周五收盘计算的，所以我们需要找到当周的所有交易日
first_week_date = pd.to_datetime(first_week['date'])
# 找到本周周一的日期
first_week_monday = first_week_date - timedelta(days=first_week_date.weekday())
# 找到本周的所有交易日
first_week_daily = daily_df[
    (pd.to_datetime(daily_df['date']) >= first_week_monday) & 
    (pd.to_datetime(daily_df['date']) <= first_week_date)
]

print("\n=== 第一周对应的日线数据 ===")
# 调整列宽确保所有列名完整显示
print("{:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<10} {:<10}".format(
    '日期', '股票代码', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额'))
print("=" * 76)
for _, row in first_week_daily.iterrows():
    # 只显示日期部分，不显示时间
    date_only = str(row['date']).split(' ')[0] if isinstance(row['date'], pd.Timestamp) else str(row['date']).split(' ')[0]
    print("{:<10} {:<8} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<10.2f} {:<10.2f}".format(
        date_only, row['code'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['amount']))
# 计算日线数据中的原始开盘价
if 'factor' in first_week_daily.columns and not first_week_daily.empty:
    first_week_daily_original_open = first_week_daily['open'].iloc[0] / first_week_daily['factor'].iloc[0]
    print(f"当周开盘价: {first_week_daily['open'].iloc[0]}")
    print(f"当周开盘价(原始价/除权价): {first_week_daily_original_open}")
else:
    print(f"当周开盘价: {first_week_daily['open'].iloc[0]}")
print(f"当周最高价: {first_week_daily['high'].max()}")
print(f"当周最低价: {first_week_daily['low'].min()}")
print(f"当周收盘价: {first_week_daily['close'].iloc[-1]}")
print(f"当周成交量: {first_week_daily['volume'].sum()}")
print(f"当周成交额: {first_week_daily['amount'].sum()}")

# 查找最后一周对应的日线数据
last_week_date = pd.to_datetime(last_week['date'])
last_week_monday = last_week_date - timedelta(days=last_week_date.weekday())
last_week_daily = daily_df[
    (pd.to_datetime(daily_df['date']) >= last_week_monday) & 
    (pd.to_datetime(daily_df['date']) <= last_week_date)
]

print("\n=== 最后一周对应的日线数据 ===")
# 调整列宽确保所有列名完整显示
print("{:<10} {:<8} {:<8} {:<8} {:<8} {:<8} {:<10} {:<10}".format(
    '日期', '股票代码', '开盘价', '最高价', '最低价', '收盘价', '成交量', '成交额'))
print("=" * 76)
for _, row in last_week_daily.iterrows():
    # 只显示日期部分，不显示时间
    date_only = str(row['date']).split(' ')[0] if isinstance(row['date'], pd.Timestamp) else str(row['date']).split(' ')[0]
    print("{:<10} {:<8} {:<8.2f} {:<8.2f} {:<8.2f} {:<8.2f} {:<10.2f} {:<10.2f}".format(
        date_only, row['code'], row['open'], row['high'], row['low'], row['close'], row['volume'], row['amount']))
# 计算最后一周日线数据中的原始开盘价
if 'factor' in last_week_daily.columns and not last_week_daily.empty:
    last_week_daily_original_open = last_week_daily['open'].iloc[0] / last_week_daily['factor'].iloc[0]
    print(f"当周开盘价: {last_week_daily['open'].iloc[0]}")
    print(f"当周开盘价(原始价/除权价): {last_week_daily_original_open}")
else:
    print(f"当周开盘价: {last_week_daily['open'].iloc[0]}")
print(f"当周最高价: {last_week_daily['high'].max()}")
print(f"当周最低价: {last_week_daily['low'].min()}")
print(f"当周收盘价: {last_week_daily['close'].iloc[-1]}")
print(f"当周成交量: {last_week_daily['volume'].sum()}")
print(f"当周成交额: {last_week_daily['amount'].sum()}")

# 关闭资源
fetcher.close()