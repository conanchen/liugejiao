import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import os
import logging
from config import STOCK_CODES, INDEX_CODES, DATA_CONFIG, LOGGING_CONFIG
# 使用qlib库替代baostock
from qlib_data_fetcher import QlibDataFetcher

# 设置中文显示
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# 配置日志
def setup_logger():
    """配置日志系统"""
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR
    }
    
    logging.basicConfig(
        level=level_map.get(LOGGING_CONFIG['level'], logging.INFO),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(LOGGING_CONFIG['file']),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger('AllTrends')

logger = setup_logger()

class BaostockDataFetcher:
    """使用 baostock 获取股票数据的类"""
    
    def __init__(self):
        """初始化 baostock 连接"""
        self.connected = False
        
    def login(self):
        """登录 baostock 服务"""
        if not self.connected:
            lg = bs.login()
            if lg.error_code != '0':
                logger.error(f"登录失败: {lg.error_msg}")
                return False
            self.connected = True
            logger.info("登录成功")
        return True
    
    def logout(self):
        """登出 baostock 服务"""
        if self.connected:
            bs.logout()
            self.connected = False
            logger.info("登出成功")
    
    def get_daily_data(self, code, start_date, end_date, adjustflag='3'):
        """
        获取股票日线数据
        
        参数:
        code: 股票代码，例如 'sh.600000'
        start_date: 开始日期，格式为 'YYYY-MM-DD'
        end_date: 结束日期，格式为 'YYYY-MM-DD'
        adjustflag: 复权类型，默认 '3' 表示前复权
                    '1': 后复权
                    '2': 不复权
                    '3': 前复权
        
        返回:
        DataFrame: 包含日线数据的 DataFrame
        """
        if not self.login():
            return None
        
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM",
            start_date=start_date,
            end_date=end_date,
            frequency="d",
            adjustflag=adjustflag
        )
        
        if rs.error_code != '0':
            logger.error(f"获取日线数据失败: {rs.error_msg}")
            return None
        
        # 解析返回数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        # 创建 DataFrame
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            # 将数值列转换为数值类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 
                              'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"成功获取 {code} 的日线数据，共 {len(df)} 条记录")
            return df
        else:
            logger.warning(f"未获取到 {code} 的数据")
            return pd.DataFrame()
    
    def get_index_daily_data(self, code, start_date, end_date):
        """
        获取指数日线数据
        
        参数:
        code: 指数代码，例如 'sh.000001'
        start_date: 开始日期，格式为 'YYYY-MM-DD'
        end_date: 结束日期，格式为 'YYYY-MM-DD'
        
        返回:
        DataFrame: 包含指数日线数据的 DataFrame
        """
        if not self.login():
            return None
        
        rs = bs.query_history_k_data_plus(
            code,
            "date,code,open,high,low,close,volume,amount,pctChg",
            start_date=start_date,
            end_date=end_date,
            frequency="d"
        )
        
        if rs.error_code != '0':
            logger.error(f"获取指数数据失败: {rs.error_msg}")
            return None
        
        # 解析返回数据
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
        
        # 创建 DataFrame
        if data_list:
            df = pd.DataFrame(data_list, columns=rs.fields)
            # 将数值列转换为数值类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'pctChg']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            logger.info(f"成功获取 {code} 的指数数据，共 {len(df)} 条记录")
            return df
        else:
            logger.warning(f"未获取到 {code} 的指数数据")
            return pd.DataFrame()
    
    def plot_price(self, df, title=None):
        """绘制价格走势图"""
        if not DATA_CONFIG['plot_charts']:
            return
        
        plt.figure(figsize=DATA_CONFIG['figure_size'])
        plt.plot(df['date'], df['close'], label='收盘价')
        plt.title(title or f"{df['code'].iloc[0]} 价格走势")
        plt.xlabel('日期')
        plt.ylabel('价格')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
    
    def calculate_ma(self, df, periods=None):
        """计算移动平均线"""
        if not periods:
            periods = DATA_CONFIG['ma_periods']
            
        for period in periods:
            df[f'ma{period}'] = df['close'].rolling(window=period).mean()
        return df
    
    def save_to_csv(self, df, filename):
        """保存数据到 CSV 文件"""
        # 确保保存目录存在
        save_dir = DATA_CONFIG['save_dir']
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # 构建完整的文件路径
        full_path = os.path.join(save_dir, filename)
        df.to_csv(full_path, index=False, encoding='utf-8-sig')
        logger.info(f"数据已保存到 {full_path}")

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
    
    df = fetcher.get_index_daily_data(index_code, start_date, end_date)
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
    
    df = fetcher.get_daily_data(stock_code, start_date, end_date, DATA_CONFIG['adjustflag'])
    if not df.empty:
        # 计算移动平均线
        if DATA_CONFIG['calculate_ma']:
            df = fetcher.calculate_ma(df)
            
            # 绘制带均线的价格图
            if DATA_CONFIG['plot_charts']:
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


if __name__ == "__main__":
    # 创建数据获取器实例 - 使用qlib替代baostock
    fetcher = QlibDataFetcher()
    
    try:
        logger.info("程序开始运行")
        
        # 初始化qlib
        if not fetcher.initialize():
            logger.error("初始化qlib失败，程序退出")
            exit(1)
            
        # 示例1: 获取所有指数数据
        fetch_all_indices(fetcher)
        
        # 示例2: 获取所有股票数据
        fetch_all_stocks(fetcher)
        
        # 如果只想获取单个股票或指数，可以使用以下方法
        # fetch_stock_data(fetcher, 'sh.600519', '贵州茅台')
        # fetch_index_data(fetcher, 'sh.000001', '上证指数')
        
    except Exception as e:
        logger.error(f"发生错误: {str(e)}", exc_info=True)
    finally:
        # 清理资源
        fetcher.close()
        logger.info("程序运行结束")