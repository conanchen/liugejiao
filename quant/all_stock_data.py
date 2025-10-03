import baostock as bs
import pandas as pd
import os

# 获取当前工作目录
base_dir = os.getcwd()
print(f"当前工作目录: {base_dir}")

#### 登陆系统 ####
lg = bs.login()
# 显示登陆返回信息
print('login respond error_code:'+lg.error_code)
print('login respond  error_msg:'+lg.error_msg)

#### 获取所有证券信息 ####
# 尝试两种方式：带日期参数和不带日期参数
print("\n=== 方式1: 带日期参数 (2024-10-25) ===")
rs_with_day = bs.query_all_stock(day="2024-10-25")
print('query_all_stock respond error_code:'+rs_with_day.error_code)
print('query_all_stock respond  error_msg:'+rs_with_day.error_msg)

# 打印结果集并保存到文件
data_list_with_day = []
while (rs_with_day.error_code == '0') & rs_with_day.next():
    # 获取一条记录，将记录合并在一起
    data_list_with_day.append(rs_with_day.get_row_data())
result_with_day = pd.DataFrame(data_list_with_day, columns=rs_with_day.fields)

# 结果集输出到csv文件 - 使用相对路径
save_file_with_day = "all_stock_with_day.csv"
result_with_day.to_csv(save_file_with_day, encoding="utf-8-sig", index=False)
print(f"方式1数据已保存到: {os.path.join(base_dir, save_file_with_day)}")
print(f"方式1获取的代码数量: {len(result_with_day)}")

print("\n=== 方式2: 不带日期参数 ===")
rs_without_day = bs.query_all_stock()
print('query_all_stock respond error_code:'+rs_without_day.error_code)
print('query_all_stock respond  error_msg:'+rs_without_day.error_msg)

# 打印结果集并保存到文件
data_list_without_day = []
while (rs_without_day.error_code == '0') & rs_without_day.next():
    # 获取一条记录，将记录合并在一起
    data_list_without_day.append(rs_without_day.get_row_data())
result_without_day = pd.DataFrame(data_list_without_day, columns=rs_without_day.fields)

# 结果集输出到csv文件 - 使用相对路径
save_file_without_day = "all_stock_without_day.csv"
result_without_day.to_csv(save_file_without_day, encoding="utf-8-sig", index=False)
print(f"方式2数据已保存到: {os.path.join(base_dir, save_file_without_day)}")
print(f"方式2获取的代码数量: {len(result_without_day)}")

# 查找是否有ETF代码
print("\n=== 检查是否存在ETF代码 (510/159开头) ===")
# 从方式2的结果中查找
if len(result_without_day) > 0:
    # 提取code列
    codes = result_without_day['code']
    # 筛选以510或159开头的代码（去掉市场前缀如sh.或sz.）
    etf_codes = []
    for code in codes:
        # 移除市场前缀
        pure_code = code.split('.')[-1]
        if pure_code.startswith('510') or pure_code.startswith('159'):
            etf_codes.append(code)
    
    print(f"找到的ETF代码数量: {len(etf_codes)}")
    if len(etf_codes) > 0:
        print("前10个ETF代码:")
        for code in etf_codes[:10]:
            print(code)
    else:
        print("未找到任何ETF代码")
else:
    print("没有获取到任何代码，无法检查ETF")

#### 登出系统 ####
bs.logout()
print('\n已登出系统')