"""
简化版Qlib $factor与原始价格关系分析

此脚本直接在控制台输出结果，演示如何使用$factor反算出原始价格。
"""

import pandas as pd
import qlib
from qlib.constant import REG_CN
from qlib.data import D

def main():
    print("===== Qlib $factor与原始价格关系分析 =====")
    
    # 初始化Qlib
    try:
        data_path = "D:/data/qlib_bin"  # Qlib数据路径
        qlib.init(provider_uri=data_path, region=REG_CN)
        print(f"✅ Qlib初始化成功，数据路径: {data_path}")
    except Exception as e:
        print(f"❌ Qlib初始化失败: {e}")
        print("请检查数据路径是否正确")
        return
    
    # 配置参数
    stocks = ["SZ000030", "SH600519","SH000300"]  # 股票代码
    start_date = "2024-01-01"  # 开始日期
    end_date = "2024-01-10"  # 结束日期
    fields = ["$open", "$high", "$low", "$close", "$factor"]  # 包含$factor字段
    
    # 获取数据
    try:
        data = D.features(
            instruments=stocks,
            fields=fields,
            start_time=start_date,
            end_time=end_date,
            freq="day"
        )
        print(f"✅ 成功获取数据，包含$factor字段: 数据形状={data.shape}")
    except Exception as e:
        print(f"❌ 获取数据失败: {e}")
        return

    # 打印数据结构信息
    print("\n===== 数据结构信息 =====")
    print(f"数据类型: {type(data)}")
    print(f"索引: {data.index.names}")
    print(f"列: {list(data.columns)}")

    # 分析$factor字段
    print("\n===== $factor字段分析 =====")

    # 获取所有股票代码
    instruments = data.index.get_level_values('instrument').unique()

    for instrument in instruments:
        # 获取单个股票的数据
        stock_data = data.loc[instrument]
        
        # 分析$factor的统计特性
        factor_min = stock_data['$factor'].min()
        factor_max = stock_data['$factor'].max()
        factor_mean = stock_data['$factor'].mean()
        factor_std = stock_data['$factor'].std()
        
        print(f"股票代码: {instrument}")
        print(f"$factor最小值: {factor_min}")
        print(f"$factor最大值: {factor_max}")
        print(f"$factor平均值: {factor_mean}")
        print(f"$factor标准差: {factor_std}")
        print(f"$factor变异系数: {factor_std/factor_mean*100:.6f}%")
        print()

    # 使用$factor反算原始价格
    print("\n===== 使用$factor反算原始价格 =====")
    print("计算方法: 原始价格 = 标准化价格 / $factor")

    # 创建一个新的DataFrame来存储原始价格
    original_data = data.copy()

    # 反算原始价格
    price_fields = ["$open", "$high", "$low", "$close"]
    for field in price_fields:
        original_data[f"original_{field[1:]}"] = original_data[field] / original_data["$factor"]

    # 显示价格对比
    print("\n===== 标准化价格与原始价格对比 =====")

    # 获取所有股票代码
    instruments = original_data.index.get_level_values('instrument').unique()

    for instrument in instruments:
        stock_data = original_data.loc[instrument]
        
        print(f"\n股票 {instrument} 价格对比:")
        
        # 显示前5行的标准化价格、$factor和原始价格
        print(f"{'日期':<12} {'开盘价':<10} {'最高价':<10} {'最低价':<10} {'收盘价':<10} {'$factor':<10} {'原始开盘价':<12} {'原始最高价':<12} {'原始最低价':<12} {'原始收盘价':<12}")
        print(f"{'-'*120}")
        
        # 打印每行数据
        for idx, row in stock_data.head().iterrows():
            print(f"{idx.strftime('%Y-%m-%d'):<12} "
                  f"{row['$open']:<10.4f} "
                  f"{row['$high']:<10.4f} "
                  f"{row['$low']:<10.4f} "
                  f"{row['$close']:<10.4f} "
                  f"{row['$factor']:<10.6f} "
                  f"{row['original_open']:<12.2f} "
                  f"{row['original_high']:<12.2f} "
                  f"{row['original_low']:<12.2f} "
                  f"{row['original_close']:<12.2f}")

    # 保存结果到CSV
    try:
        # 扁平化数据，便于保存和查看
        flat_data = original_data.reset_index()
        csv_file = "original_prices.csv"
        flat_data.to_csv(csv_file, index=False, encoding="utf-8")
        print(f"\n✅ 原始价格数据已保存到: {csv_file}")
    except Exception as e:
        print(f"❌ 保存数据失败: {e}")

    # 总结
    print("\n===== $factor与原始价格关系总结 =====")
    print("1. 在Qlib中，$factor通常用于复权计算")
    print("2. 对于同一支股票，$factor值通常保持相对稳定，只有在除权除息时才会发生变化")
    print("3. 原始价格的计算公式: 原始价格 = 标准化价格 / $factor")
    print("4. 标准化价格是经过复权处理后的价格，便于进行技术分析和回测")
    print("5. 通过$factor可以将标准化价格反算回原始市场价格")

    print("\n✅ 分析完成")

# Windows多进程兼容：确保代码只在主进程中执行
if __name__ == "__main__":
    main()