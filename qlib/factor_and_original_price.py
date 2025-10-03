"""
Qlib中$factor与原始价格关系分析

此脚本详细分析Qlib中$factor字段的作用，并展示如何使用它来反算出原始价格。
"""

import pandas as pd
import logging
import sys
import os
import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 配置日志
def setup_logging():
    # 确保日志目录存在
    log_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 配置日志，同时输出到文件和控制台
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(os.path.join(log_dir, 'factor_analysis.log'), encoding='utf-8'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logging.info("日志配置完成")

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

# 获取包含$factor字段的数据
def get_data_with_factor(instruments, start_time, end_time, freq="day"):
    """获取包含$factor字段的数据"""
    fields = ["$open", "$high", "$low", "$close", "$volume", "$factor"]
    
    try:
        data = D.features(
            instruments=instruments,
            fields=fields,
            start_time=start_time,
            end_time=end_time,
            freq=freq
        )
        logging.info(f"成功获取数据，包含$factor字段: 数据形状={data.shape}")
        return data
    except Exception as e:
        logging.error(f"获取数据失败: {e}")
        return None

# 分析$factor字段
def analyze_factor(data):
    """分析$factor字段的特性"""
    if data is None or data.empty:
        logging.warning("没有数据可分析")
        return
    
    logging.info("\n===== $factor字段分析 =====")
    
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
        
        logging.info(f"股票代码: {instrument}")
        logging.info(f"$factor最小值: {factor_min}")
        logging.info(f"$factor最大值: {factor_max}")
        logging.info(f"$factor平均值: {factor_mean}")
        logging.info(f"$factor标准差: {factor_std}")
        logging.info(f"$factor变异系数: {factor_std/factor_mean*100:.6f}%")
        logging.info("")

# 使用$factor反算原始价格
def calculate_original_prices(data):
    """使用$factor反算原始价格"""
    if data is None or data.empty:
        logging.warning("没有数据可计算")
        return None
    
    # 创建一个新的DataFrame来存储原始价格
    original_data = data.copy()
    
    # 反算原始价格
    price_fields = ["$open", "$high", "$low", "$close"]
    
    # 注意：这里使用除法还是乘法取决于数据预处理的方式
    # 在Qlib中，通常的做法是：原始价格 = 标准化价格 / $factor
    for field in price_fields:
        original_data[f"original_{field[1:]}"] = original_data[field] / original_data["$factor"]
    
    logging.info("\n===== 使用$factor反算原始价格 =====")
    logging.info("计算方法: 原始价格 = 标准化价格 / $factor")