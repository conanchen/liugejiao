# @brief: 使用微软qlib库获取股票数据，替代baostock
# 数据目录为d:/data/qlib_bin

import qlib
from qlib.data import D
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
from config import STOCK_CODES, INDEX_CODES, DATA_CONFIG, LOGGING_CONFIG

# 配置日志
logger = logging.getLogger('QlibDataFetcher')

class QlibDataFetcher:
    """使用 qlib 获取股票数据的类"""
    
    def __init__(self):
        """初始化 qlib 连接"""
        self.initialized = False
        self.provider_uri = "d:/data/qlib_bin"
        
    def initialize(self):
        """初始化 qlib 数据"""
        if not self.initialized:
            try:
                qlib.init(provider_uri=self.provider_uri, region='cn')
                self.initialized = True
                logger.info(f"成功初始化qlib，数据路径: {self.provider_uri}")
            except Exception as e:
                logger.error(f"初始化qlib失败: {str(e)}")
                return False
        return True
    
    def get_daily_data(self, code, start_date, end_date):
        """
        获取通用日线数据，适用于股票、指数、ETF等各类金融产品
        
        参数:
        code: 代码，例如 'SH600000'、'SH000001' 或'SH510300'
        start_date: 开始日期，格式为 'YYYY-MM-DD'
        end_date: 结束日期，格式为 'YYYY-MM-DD'
        
        返回:
        DataFrame: 包含日线数据的 DataFrame
        
        备注：qlib 数据已经是后复权的，无需额外处理复权
        """
        if not self.initialize():
            return None
        
        try:
            qlib_code = code
            
            # 使用通用字段集合，适合各类金融产品
            fields = ['$open', '$high', '$low', '$close', '$volume', '$amount']
            
            # 尝试获取数据
            try:
                df = D.features([qlib_code], fields, start_date, end_date)
            except Exception as e:
                logger.error(f"获取 {code} 的日线数据失败: {str(e)}")
                return None
            
            if df.empty:
                logger.warning(f"未获取到 {code} 的日线数据")
                return pd.DataFrame()
            
            # 重置索引并处理数据格式
            df = df.reset_index()
            
            # 重命名列
            rename_dict = {
                'datetime': 'date',
                '$open': 'open',
                '$high': 'high',
                '$low': 'low',
                '$close': 'close',
                '$volume': 'volume',
                '$amount': 'amount'
            }
            df = df.rename(columns=rename_dict)
            
            # 保持原始代码格式
            df['code'] = code
            
            # 计算涨跌幅
            df['pctChg'] = df['close'].pct_change() * 100
            
            # 处理数值列
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
            for col in numeric_columns:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 确保date列是datetime类型
            df['date'] = pd.to_datetime(df['date'])
            
            logger.info(f"成功获取 {code} 的日线数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取 {code} 的日线数据失败: {str(e)}")
            return None
    

    
    def get_weekly_data(self, code, start_date, end_date):
        """
        获取股票周线数据
        
        参数:
        code: 股票代码，例如 'SH600000'
        start_date: 开始日期，格式为 'YYYY-MM-DD'
        end_date: 结束日期，格式为 'YYYY-MM-DD'
        
        返回:
        DataFrame: 包含周线数据的 DataFrame
        
        备注：qlib 数据已经是后复权的，无需额外处理复权
        """
        # 获取日线数据，然后转换为周线
        daily_df = self.get_daily_data(code, start_date, end_date)
        
        if daily_df is None or daily_df.empty:
            return None
        
        try:
            # 设置日期索引
            daily_df = daily_df.set_index('date')
            
            # 转换为周线数据
            weekly_df = daily_df.resample('W-FRI').agg({
                'code': 'last',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum',
                'amount': 'sum'
            })
            
            # 重置索引
            weekly_df = weekly_df.reset_index()
            
            # 计算周涨跌幅
            weekly_df['pctChg'] = weekly_df['close'].pct_change() * 100
            
            logger.info(f"成功获取 {code} 的周线数据，共 {len(weekly_df)} 条记录")
            return weekly_df
            
        except Exception as e:
            logger.error(f"转换周线数据失败: {str(e)}")
            return None
    

    
    def calculate_ma(self, df, periods=None):
        """计算移动平均线，与BaostockDataFetcher保持一致的接口"""
        if not periods:
            periods = DATA_CONFIG['ma_periods']
            
        for period in periods:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def plot_price(self, df, title=None):
        """绘制价格走势图，与BaostockDataFetcher保持一致的接口"""
        # 这里可以保持与原来相同的实现
        # 导入matplotlib并绘制图表
        import matplotlib.pyplot as plt
        
        if not DATA_CONFIG['plot_charts']:
            return
        
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        plt.figure(figsize=DATA_CONFIG['figure_size'])
        plt.plot(df['date'], df['close'], label='收盘价')
        plt.title(title or f"{df['code'].iloc[0]} 价格走势")
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def save_to_csv(self, df, filename):
        """保存数据到 CSV 文件，与BaostockDataFetcher保持一致的接口"""
        # 确保保存目录存在
        save_dir = DATA_CONFIG['save_dir']
        import os
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 构建完整的文件路径
        full_path = os.path.join(save_dir, filename)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存到 {full_path}")
    
    def close(self):
        """清理资源"""
        # qlib不需要显式登出，这里保持接口一致性
        logger.info("qlib资源已清理")

# 兼容原有的函数接口

def get_date_range():
    """根据配置获取日期范围"""
    if DATA_CONFIG['use_relative_dates']:
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=DATA_CONFIG['days_back'])).strftime('%Y-%m-%d')
    else:
        start_date = DATA_CONFIG['start_date']
        end_date = DATA_CONFIG['end_date']
    return start_date, end_date

def fetch_index_data(fetcher, index_code, index_name):
    """获取指定指数的数据"""
    logger.info(f"获取 {index_name} ({index_code}) 数据...")
    start_date, end_date = get_date_range()
    
    df = fetcher.get_daily_data(index_code, start_date, end_date)
    if not df.empty:
        # 绘制价格走势图
        fetcher.plot_price(df, title=f"{index_name} ({index_code}) 价格走势")
        
        # 保存数据
        filename = f"{index_code}_index_data.csv"
        fetcher.save_to_csv(df, filename)
    return df

def fetch_stock_data(fetcher, stock_code, stock_name):
    """获取指定股票的数据"""
    logger.info(f"获取 {stock_name} ({stock_code}) 数据...")
    start_date, end_date = get_date_range()
    
    df = fetcher.get_daily_data(stock_code, start_date, end_date)
    if not df.empty:
        # 计算移动平均线
        if DATA_CONFIG['calculate_ma']:
            df = fetcher.calculate_ma(df)
            
            # 绘制带均线的价格图
            if DATA_CONFIG['plot_charts']:
                import matplotlib.pyplot as plt
                plt.rcParams['font.sans-serif'] = ['SimHei']
                plt.rcParams['axes.unicode_minus'] = False
                
                plt.figure(figsize=DATA_CONFIG['figure_size'])
                plt.plot(df['date'], df['close'], label='收盘价')
                for period in DATA_CONFIG['ma_periods']:
                    plt.plot(df['date'], df[f'ma{period}'], label=f'ma{period}')
                plt.title(f"{stock_name} ({stock_code}) 价格及均线走势")
                plt.xlabel('日期')
                plt.ylabel('价格')
                plt.legend()
                plt.grid(True)
                plt.tight_layout()
                plt.show()
        
        # 保存数据
        filename = f"{stock_code}_stock_data.csv"
        fetcher.save_to_csv(df, filename)
    return df

def fetch_all_stocks(fetcher):
    """获取配置中的所有股票数据"""
    for stock_code, stock_name in STOCK_CODES.items():
        fetch_stock_data(fetcher, stock_code, stock_name)

def fetch_all_indices(fetcher):
    """获取配置中的所有指数数据"""
    for index_code, index_name in INDEX_CODES.items():
        fetch_index_data(fetcher, index_code, index_name)

# 示例用法
if __name__ == "__main__":
    # 设置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('qlib_fetcher.log'),
            logging.StreamHandler()
        ]
    )
    
    # 创建qlib数据获取器
    fetcher = QlibDataFetcher()
    
    try:
        # 测试获取单个股票数据
        stock_code = 'SH600519'  # 贵州茅台
        stock_name = '贵州茅台'
        df = fetch_stock_data(fetcher, stock_code, stock_name)
        print(f"获取到 {stock_code} 的数据行数: {len(df) if df is not None else 0}")
        
        # 测试获取单个指数数据
        index_code = 'SH000001'  # 上证指数
        index_name = '上证指数'
        df_index = fetch_index_data(fetcher, index_code, index_name)
        print(f"获取到 {index_code} 的数据行数: {len(df_index) if df_index is not None else 0}")
        
        # 测试根据日线数据获取周线数据
        start_date, end_date = get_date_range()
        df_week = fetcher.get_weekly_data(stock_code, start_date, end_date)
        print(f"获取到 {stock_code} 的周线数据行数: {len(df_week) if df_week is not None else 0}")
        
    except Exception as e:
        logging.error(f"发生错误: {str(e)}", exc_info=True)
    finally:
        fetcher.close()