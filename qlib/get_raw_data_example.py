"""
Qlib获取股票原始数据示例脚本

此脚本展示如何使用Qlib库获取和处理股票原始数据。
注意：在Qlib中，"原始数据"通常指的是经过基本清洗但未标准化的市场数据。
"""

import pandas as pd
import logging
import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 配置日志
def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('get_raw_data.log', encoding='utf-8'),
            logging.StreamHandler()
        ]
    )

# 初始化Qlib
def init_qlib(data_path="D:/data/qlib_bin"):
    """初始化Qlib库"""
    try:
        qlib.init(provider_uri=data_path, region=REG_CN)
        logging.info(f"Qlib初始化成功，数据路径: {data_path}")
        return True
    except Exception as e:
        logging.error(f"Qlib初始化失败: {e}")
        return False

# 获取原始市场数据
def get_market_data(
    instruments, 
    fields, 
    start_time, 
    end_time,
    freq="day"
):
    """
    获取股票市场数据
    
    参数:
    - instruments: 股票代码列表，如["SZ000030", "SH600519"]
    - fields: 字段列表，如["$open", "$high", "$low", "$close", "$volume"]
    - start_time: 开始时间，如"2024-01-01"
    - end_time: 结束时间，如"2024-01-10"
    - freq: 数据频率，如"day"(日线), "1min"(1分钟线)等
    
    返回:
    - pandas DataFrame格式的数据
    """
    try:
        # 使用D.features获取数据
        data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_time,
            end_time=end_time,
            freq=freq
        )
        logging.info(f"成功获取数据: 股票数量={len(instruments)}, 字段数量={len(fields)}, 数据形状={data.shape}")
        return data
    except Exception as e:
        logging.error(f"获取数据失败: {e}")
        return None

# 由于Qlib版本差异，get_field函数在某些版本中可能不可用
# 可以通过D.features获取单个字段的数据，如下所示

# 查看数据结构
def inspect_data(data, title="数据样例"):
    """查看数据结构和内容"""
    if data is None or data.empty:
        logging.warning("没有数据可查看")
        return
    
    logging.info(f"\n===== {title} =====")
    logging.info(f"数据类型: {type(data)}")
    logging.info(f"数据形状: {data.shape}")
    logging.info(f"数据索引: {data.index.names}")
    logging.info(f"数据列: {list(data.columns)}")
    logging.info("")
    
    # 显示数据类型
    logging.info(f"数据字段类型:\n{data.dtypes}")
    logging.info("")
    
    # 显示前5行数据
    logging.info(f"数据前5行:\n{data.head()}\n")

# 主函数
def main():
    setup_logging()
    
    # 初始化Qlib
    if not init_qlib():
        return
    
    # 配置参数
    stocks = ["SZ000030", "SH600519"]  # 股票代码
    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]  # 字段
    start_date = "2024-01-01"  # 开始日期
    end_date = "2024-01-10"  # 结束日期
    
    # 示例1: 获取多个股票的多个字段数据
    logging.info("\n=== 示例1: 获取多个股票的多个字段数据 ===")
    market_data = get_market_data(stocks, fields, start_date, end_date)
    inspect_data(market_data, "多股票多字段数据")
    
    # 示例2: 获取单个股票的单个字段数据
    logging.info("\n=== 示例2: 获取单个股票的单个字段数据 ===")
    # 使用D.features获取单个字段数据
    close_price = get_market_data(["SZ000030"], ["$close"], start_date, end_date)
    inspect_data(close_price, "单个股票收盘价数据")
    
    # 示例3: 处理和转换数据
    logging.info("\n=== 示例3: 处理和转换数据 ===")
    if market_data is not None and not market_data.empty:
        # 转换为传统的表格格式（如果需要）
        flat_data = market_data.reset_index()
        logging.info(f"扁平化后的数据形状: {flat_data.shape}")
        logging.info(f"扁平化后的数据前5行:\n{flat_data.head()}\n")
        
        # 保存为CSV文件
        csv_file = "qlib_market_data.csv"
        flat_data.to_csv(csv_file, index=False, encoding="utf-8")
        logging.info(f"数据已保存到CSV文件: {csv_file}")
    
    logging.info("\n=== Qlib数据说明 ===")
    logging.info("1. 在Qlib中，数据通常以pandas的MultiIndex DataFrame格式存储")
    logging.info("2. 一级索引是instrument(股票代码)，二级索引是datetime(日期时间)")
    logging.info("3. 列是各个字段，如开盘价、最高价、最低价、收盘价、成交量等")
    logging.info("4. Qlib中的数据可能经过了时区调整和基本的清洗")
    logging.info("5. 对于需要完全原始数据的场景，可能需要从数据源直接获取或使用专门的数据接口")

if __name__ == "__main__":
    main()