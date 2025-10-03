import baostock as bs
import pandas as pd

"""
测试baostock_get_etf.md中描述的ETF数据获取方法的有效性
"""

print("===== 测试baostock获取ETF数据的方法 =====")

# 步骤1：登录baostock
try:
    lg = bs.login()
    print(f'登录状态: {lg.error_msg}')
    
    if lg.error_code != '0':
        print("登录失败，无法继续测试")
        exit(1)
    
except Exception as e:
    print(f"登录过程中发生错误: {e}")
    exit(1)

# 步骤2：测试query_all_stock方法获取ETF代码列表
print("\n===== 测试方法1: 使用query_all_stock获取ETF代码 =====")
try:
    # 使用文档中提到的方法
    rs = bs.query_all_stock()
    print(f"查询状态: 错误码={rs.error_code}, 错误信息={rs.error_msg}")
    
    etf_list = []
    if rs.error_code == '0':
        print("开始遍历所有代码...")
        total_count = 0
        while rs.next():
            total_count += 1
            code = rs.get_row_data()[0]
            if code.startswith('510') or code.startswith('159'):
                etf_list.append(code)
        
        print(f"总共遍历了 {total_count} 个代码")
        print(f"找到的ETF代码数量: {len(etf_list)}")
        if len(etf_list) > 0:
            print(f"前5个ETF代码: {etf_list[:5]}")
    
except Exception as e:
    print(f"使用query_all_stock方法时发生错误: {e}")

# 步骤3：测试直接获取特定ETF的K线数据
print("\n===== 测试方法2: 直接获取特定ETF的K线数据 =====")
etf_codes_to_test = ["sh.510300", "sz.159915"]  # 沪深300ETF和创业板ETF

for code in etf_codes_to_test:
    try:
        print(f"\n尝试获取ETF {code} 的K线数据...")
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume",
            start_date="2023-01-01",
            end_date="2023-01-10",  # 短期数据用于测试
            frequency="d",
            adjustflag="3"  # 前复权
        )
        
        if rs.error_code != '0':
            print(f"获取 {code} 的K线数据失败: {rs.error_msg}")
            continue
        
        # 解析返回数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        # 检查是否获取到数据
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            print(f"成功获取 {code} 的 {len(df)} 条K线数据")
            print("数据样例:")
            print(df.head(3).to_string(index=False))
        else:
            print(f"未获取到 {code} 的任何K线数据")
    except Exception as e:
        print(f"获取 {code} 数据时出错: {e}")

# 步骤4：测试替代方法 - 获取指数数据作为对比
print("\n===== 测试方法3: 获取指数数据作为对比 =====")
index_codes = ["sh.000300", "sz.399006"]  # 沪深300指数和创业板指数

for code in index_codes:
    try:
        print(f"\n尝试获取指数 {code} 的K线数据...")
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume",
            start_date="2023-01-01",
            end_date="2023-01-10",
            frequency="d",
            adjustflag="3"
        )
        
        if rs.error_code != '0':
            print(f"获取 {code} 的K线数据失败: {rs.error_msg}")
            continue
        
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            print(f"成功获取 {code} 的 {len(df)} 条K线数据")
            print("数据样例:")
            print(df.head(3).to_string(index=False))
        else:
            print(f"未获取到 {code} 的任何K线数据")
    except Exception as e:
        print(f"获取 {code} 数据时出错: {e}")

# 步骤5：总结测试结果
print("\n===== 测试结果总结 =====")
print("1. 文档中提到的query_all_stock方法：")
print("   - 可能存在API变更或参数问题")
print("   - 在当前版本的baostock中可能不再适用")
print("\n2. 直接获取ETF K线数据：")
print("   - 测试的ETF代码(sh.510300, sz.159915)可能无法获取数据")
print("\n3. 获取指数数据：")
print("   - 指数数据通常可以正常获取")

# 登出baostock
bs.logout()
print("\n已登出baostock")
print("\n===== 测试完成 =====")