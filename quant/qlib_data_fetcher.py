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
        
        # 技术指标配置
        self.calculate_indicators = True  # 默认计算技术指标
        self.ma_periods = DATA_CONFIG.get('ma_periods', [5, 10, 20, 60])  # 均线周期
        self.rsi_period = 14  # RSI周期
        self.macd_fast = 12  # MACD快线周期
        self.macd_slow = 26  # MACD慢线周期
        self.macd_signal = 9  # MACD信号线周期
        self.bollinger_period = 20  # 布林带周期
        self.bollinger_std = 2  # 布林带标准差倍数
        self.volatility_period = 20  # 波动率计算周期
        
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
    
    def get_daily_data(self, code, start_date, end_date, calculate_indicators=None):
        """
        获取通用日线数据，适用于股票、指数、ETF等各类金融产品
        
        参数:
        code: 代码，例如 'SH600000'、'SH000001' 或'SH510300'
        start_date: 开始日期，格式为 'YYYY-MM-DD'
        end_date: 结束日期，格式为 'YYYY-MM-DD'
        calculate_indicators: 是否计算技术指标，默认为None(使用实例设置)
        
        返回:
        DataFrame: 包含日线数据和技术指标的 DataFrame
        
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
            
            # 根据参数决定是否计算技术指标
            if calculate_indicators or (calculate_indicators is None and self.calculate_indicators):
                df = self.calculate_technical_indicators(df)
            
            logger.info(f"成功获取 {code} 的日线数据，共 {len(df)} 条记录")
            return df
            
        except Exception as e:
            logger.error(f"获取 {code} 的日线数据失败: {str(e)}")
            return None
    
    def calculate_technical_indicators(self, df):
        """计算各种技术指标"""
        try:
            # 计算移动平均线
            for period in self.ma_periods:
                df[f'ma{period}'] = df['close'].rolling(window=period).mean()
            
            # 计算RSI指标
            delta = df['close'].diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
            rs = gain / loss
            df['rsi'] = 100 - (100 / (1 + rs))
            
            # 计算MACD指标
            df['ema_fast'] = df['close'].ewm(span=self.macd_fast, adjust=False).mean()
            df['ema_slow'] = df['close'].ewm(span=self.macd_slow, adjust=False).mean()
            df['macd'] = df['ema_fast'] - df['ema_slow']
            df['macd_signal'] = df['macd'].ewm(span=self.macd_signal, adjust=False).mean()
            df['macd_hist'] = df['macd'] - df['macd_signal']
            
            # 计算布林带
            df['bollinger_mid'] = df['close'].rolling(window=self.bollinger_period).mean()
            df['bollinger_std'] = df['close'].rolling(window=self.bollinger_period).std()
            df['bollinger_upper'] = df['bollinger_mid'] + (df['bollinger_std'] * self.bollinger_std)
            df['bollinger_lower'] = df['bollinger_mid'] - (df['bollinger_std'] * self.bollinger_std)
            
            # 计算波动率 (ATR)
            df['high_low'] = df['high'] - df['low']
            df['high_close'] = np.abs(df['high'] - df['close'].shift(1))
            df['low_close'] = np.abs(df['low'] - df['close'].shift(1))
            df['tr'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
            df['atr'] = df['tr'].rolling(window=self.volatility_period).mean()
            
            # 删除临时列
            temp_columns = ['high_low', 'high_close', 'low_close', 'tr', 'ema_fast', 'ema_slow']
            df = df.drop(columns=[col for col in temp_columns if col in df.columns])
            
            logger.info(f"成功计算技术指标: 均线{self.ma_periods}, RSI{self.rsi_period}, MACD({self.macd_fast},{self.macd_slow},{self.macd_signal}), 布林带({self.bollinger_period}), ATR({self.volatility_period})")
        except Exception as e:
            logger.error(f"计算技术指标失败: {str(e)}")
        
        return df
    
    def get_weekly_data(self, code, start_date, end_date, calculate_indicators=None):
        """
        获取股票周线数据
        
        参数:
        code: 股票代码，例如 'SH600000'
        start_date: 开始日期，格式为 'YYYY-MM-DD'
        end_date: 结束日期，格式为 'YYYY-MM-DD'
        calculate_indicators: 是否计算技术指标，默认为None(使用实例设置)
        
        返回:
        DataFrame: 包含周线数据和技术指标的 DataFrame
        
        备注：qlib 数据已经是后复权的，无需额外处理复权
        """
        # 获取日线数据，然后转换为周线
        daily_df = self.get_daily_data(code, start_date, end_date, calculate_indicators=False)  # 先不计算指标，等转换为周线后再计算
        
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
            
            # 根据参数决定是否计算技术指标
            if calculate_indicators or (calculate_indicators is None and self.calculate_indicators):
                weekly_df = self.calculate_technical_indicators(weekly_df)
            
            logger.info(f"成功获取 {code} 的周线数据，共 {len(weekly_df)} 条记录")
            return weekly_df
            
        except Exception as e:
            logger.error(f"转换周线数据失败: {str(e)}")
            return None
    

    
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
    
    # 获取日线数据（包含技术指标）
    df = fetcher.get_daily_data(stock_code, start_date, end_date)
    if not df.empty:
        # 绘制带均线的价格图
        if DATA_CONFIG.get('plot_charts', False):
            import matplotlib.pyplot as plt
            plt.rcParams['font.sans-serif'] = ['SimHei']
            plt.rcParams['axes.unicode_minus'] = False
            
            plt.figure(figsize=DATA_CONFIG.get('figure_size', (12, 6)))
            plt.plot(df['date'], df['close'], label='收盘价')
            periods = DATA_CONFIG.get('ma_periods', [5, 10, 20, 60])
            for period in periods:
                if f'ma{period}' in df.columns:
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