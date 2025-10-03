import baostock as bs
import pandas as pd

"""
检查baostock是否提供ETF数据的脚本
功能：
1. 尝试直接获取ETF的K线数据
2. 验证配置文件中指定的ETF代码是否可用
3. 与指数数据获取进行对比
"""

print("===== 检查baostock ETF数据可用性 =====")

# 登录baostock
try:
    lg = bs.login()
    print(f'登录状态: {lg.error_msg}')
    
    if lg.error_code != '0':
        print("登录失败，无法继续检查")
        exit(1)
    
except Exception as e:
    print(f"登录过程中发生错误: {e}")
    exit(1)

print("\n===== 从配置文件中获取的ETF和指数代码 =====")
print("ETF代码: sh.510300 (沪深300ETF)")
print("指数代码: sh.000300 (沪深300指数)")

print("\n===== 测试获取ETF的K线数据 =====")

# 测试函数：获取K线数据
def test_fetch_k_data(code, name, start_date, end_date):
    """测试获取指定代码的K线数据"""
    try:
        print(f"\n尝试获取 {name} ({code}) 的数据...")
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag="3"  # 前复权
        )
        
        if rs.error_code != '0':
            print(f"获取 {code} 的K线数据失败: {rs.error_msg}")
            return None
        
        # 解析返回数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        # 检查是否获取到数据
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            print(f"成功获取 {code} 的 {len(df)} 条K线数据")
            # 显示最近5条数据
            if len(df) >= 5:
                print("\n最近5条K线数据:")
                print(df.tail(5).to_string(index=False))
            return df
        else:
            print(f"未获取到 {code} 的任何K线数据")
            return None
    except Exception as e:
        print(f"获取 {code} 数据时出错: {e}")
        return None

# 测试多个时间段以增加成功率
time_ranges = [
    ("2024-09-01", "2024-09-30"),  # 最近一个月
    ("2024-01-01", "2024-09-30"),  # 今年以来
    ("2023-01-01", "2023-12-31")   # 去年全年
]

# 测试ETF数据
etf_code = "sh.510300"
etf_name = "沪深300ETF"
etf_success = False

for start_date, end_date in time_ranges:
    print(f"\n测试时间段: {start_date} 至 {end_date}")
    df = test_fetch_k_data(etf_code, etf_name, start_date, end_date)
    if df is not None and len(df) > 0:
        etf_success = True
        # 保存数据到CSV文件以便查看
        df.to_csv(f"{etf_code}_test_data.csv", index=False, encoding="utf-8-sig")
        print(f"\n数据已保存到 {etf_code}_test_data.csv 文件")
        break

# 测试指数数据（作为对比）
index_code = "sh.000300"
index_name = "沪深300指数"

print(f"\n===== 作为对比，测试获取{index_name}的K线数据 =====")
index_df = test_fetch_k_data(index_code, index_name, time_ranges[0][0], time_ranges[0][1])
if index_df is not None and len(index_df) > 0:
    index_df.to_csv(f"{index_code}_comparison.csv", index=False, encoding="utf-8-sig")
    print(f"\n指数数据已保存到 {index_code}_comparison.csv 文件")

# 总结结果
print("\n===== 检查结果总结 =====")
if etf_success:
    print(f"✓ 成功获取 {etf_name} ({etf_code}) 的数据")
else:
    print(f"✗ 无法获取 {etf_name} ({etf_code}) 的数据")
    print("  这可能意味着baostock不支持该ETF代码的数据查询")
    
if index_df is not None and len(index_df) > 0:
    print(f"✓ 成功获取 {index_name} ({index_code}) 的数据")
else:
    print(f"✗ 无法获取 {index_name} ({index_code}) 的数据")

print("\n建议：")
if not etf_success and index_df is not None and len(index_df) > 0:
    print(f"1. 考虑使用 {index_name} ({index_code}) 替代 {etf_name} ({etf_code})")
    print("2. 在您的策略中，您可以继续使用当前的指数数据配置")

# 登出baostock
bs.logout()
print("\n已登出baostock")
print("\n===== 检查完成 =====")