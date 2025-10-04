#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import sys
import qlib
from qlib.data import D
import pandas as pd
import numpy as np

# 禁用多进程，避免Windows环境下的问题
sys.setrecursionlimit(1000000)
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["WINDOWS_DISABLE_IPC"] = "1"

# 初始化Qlib
def init_qlib():
    """初始化Qlib环境，使用用户提供的数据路径"""
    try:
        # 用户提供的Qlib数据路径
        data_path = "D:/data/qlib_bin"
        
        if os.path.exists(data_path):
            print(f"使用用户提供的数据路径: {data_path}")
            qlib.init(provider_uri=data_path)
            print("✅ Qlib初始化成功")
        else:
            raise Exception(f"用户提供的数据路径不存在: {data_path}")
    except Exception as e:
        print(f"❌ Qlib初始化失败: {str(e)}")
        sys.exit(1)

def 验证Qlib复权方式(股票代码="SH600519", 开始日期="2005-01-04", 结束日期="2025-09-30"):
    """
    验证Qlib使用的是前复权还是后复权
    用户从deepseek复制的代码
    """
    # 获取Qlib数据
    data = D.features(
        instruments=[股票代码],
        fields=['$open', '$close', '$high', '$low', '$volume', '$factor'],
        start_time=开始日期,
        end_time=结束日期
    )
    
    # 计算除权价
    data['计算除权价'] = data['$close'] / data['$factor']
    
    # 获取最新复权因子（通常是1.0）
    最新复权因子 = data['$factor'].iloc[-1]
    
    # 计算前复权价价格（保持最新价格不变）
    data['计算前复权价'] = data['计算除权价'] * (最新复权因子 / data['$factor'])
    
    # 计算后复权价价格（保持历史价格不变）
    data['计算后复权价'] = data['计算除权价'] * (data['$factor'].iloc[0] / data['$factor'])
    
    print("=== Qlib复权方式验证 ===")
    print(f"股票: {股票代码}")
    print(f"数据期间: {开始日期} 到 {结束日期}")
    print(f"最新复权因子: {最新复权因子:.6f}")
    print(f"最初复权因子: {data['$factor'].iloc[0]:.6f}")
    print()
    
    # 处理元组类型的索引
    # Qlib的数据索引通常是(instrument, datetime)的元组
    print("数据索引类型: ", type(data.index[0]))
    print("数据索引示例: ", data.index[0])
    
    # 比较Qlib提供的数据与我们计算的数据
    print("\n前5个交易日对比:")
    print("日期 | Qlib收盘价 | 前复权价 | 后复权价 | 除权价 | 复权因子")
    for i in range(min(5, len(data))):
        # 获取索引元组
        index_tuple = data.index[i]
        # 如果是元组，取第二个元素作为日期
        if isinstance(index_tuple, tuple):
            日期 = index_tuple[1]  # 假设第二个元素是日期
        else:
            日期 = index_tuple
        
        qlib价格 = data['$close'].iloc[i]
        计算前复权价 = data['计算前复权价'].iloc[i]
        计算后复权价 = data['计算后复权价'].iloc[i]
        计算除权价 = data['计算除权价'].iloc[i]
        复权因子 = data['$factor'].iloc[i]
        
        # 确保日期可以格式化
        if hasattr(日期, 'strftime'):
            日期_str = 日期.strftime('%Y-%m-%d')
        else:
            日期_str = str(日期)
            
        print(f"{日期_str} | {qlib价格:.2f} | {计算前复权价:.2f} | {计算后复权价:.2f} | {计算除权价:.2f} | {复权因子:.6f}")
    
    print()
    print("最后5个交易日对比:")
    for i in range(max(0, len(data)-5), len(data)):
        # 获取索引元组
        index_tuple = data.index[i]
        # 如果是元组，取第二个元素作为日期
        if isinstance(index_tuple, tuple):
            日期 = index_tuple[1]  # 假设第二个元素是日期
        else:
            日期 = index_tuple
        
        qlib价格 = data['$close'].iloc[i]
        计算前复权价 = data['计算前复权价'].iloc[i]
        计算后复权价 = data['计算后复权价'].iloc[i]
        计算除权价 = data['计算除权价'].iloc[i]
        复权因子 = data['$factor'].iloc[i]
        
        # 确保日期可以格式化
        if hasattr(日期, 'strftime'):
            日期_str = 日期.strftime('%Y-%m-%d')
        else:
            日期_str = str(日期)
            
        print(f"{日期_str} | {qlib价格:.2f} | {计算前复权价:.2f} | {计算后复权价:.2f} | {计算除权价:.2f} | {复权因子:.6f}")
    
    # 判断Qlib使用的是哪种复权
    qlib与前复权差异 = abs(data['$close'] - data['计算前复权价']).mean()
    qlib与后复权差异 = abs(data['$close'] - data['计算后复权价']).mean()
    
    print()
    print("=== 判断结果 ===")
    print(f"Qlib价格 vs 前复权价 平均差异: {qlib与前复权差异:.6f}")
    print(f"Qlib价格 vs 后复权价 平均差异: {qlib与后复权差异:.6f}")
    
    # 额外添加对前复权计算方式的验证
    # 方法1：使用最新复权因子保持最新价格不变
    data['前复权价_方法1'] = data['计算除权价'] * (最新复权因子 / data['$factor'])
    
    # 方法2：使用相邻日期的复权因子计算连续前复权
    data['前复权价_方法2'] = data['计算除权价'] * (data['$factor'] / data['$factor'].iloc[0])
    
    # 计算各种方法的差异
    方法1与Qlib差异 = abs(data['$close'] - data['前复权价_方法1']).mean()
    方法2与Qlib差异 = abs(data['$close'] - data['前复权价_方法2']).mean()
    
    print()
    print("=== 不同前复权计算方法对比 ===")
    print(f"方法1 (最新因子) 与Qlib差异: {方法1与Qlib差异:.6f}")
    print(f"方法2 (连续因子) 与Qlib差异: {方法2与Qlib差异:.6f}")
    
    # 保存验证结果到CSV文件
    result_df = data.copy()
    result_df.to_csv("verify_qlib_adj_method.csv", encoding="utf-8-sig")
    print(f"\n✅ 验证结果已保存到 verify_qlib_adj_method.csv")
    
    if qlib与前复权差异 < qlib与后复权差异:
        print("✅ Qlib使用的是: 前复权")
        return "前复权"
    else:
        print("✅ Qlib使用的是: 后复权")  
        return "后复权"

if __name__ == "__main__":
    # 初始化Qlib
    init_qlib()
    
    # 运行验证
    print("\n开始验证Qlib复权方式...")
    验证结果 = 验证Qlib复权方式()
    
    # 添加更多验证场景
    print("\n使用不同时间段进行额外验证...")
    验证结果2 = 验证Qlib复权方式(开始日期="2024-01-04", 结束日期="2024-09-30")
    
    print("\n=== 最终结论 ===")
    if 验证结果 == 验证结果2 == "后复权":
        print("✅ 通过多重验证确认：Qlib默认提供的$open/$close等价格字段确实是后复权价格。")
    else:
        print("❓ 验证结果不一致，请检查数据和计算方法。")