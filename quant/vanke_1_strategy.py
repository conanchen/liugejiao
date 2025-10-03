import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import warnings
from datetime import datetime
import baostock as bs

# 只禁用字体相关警告
warnings.filterwarnings("ignore", category=UserWarning, module="matplotlib.font_manager")

# 简化字体设置，只使用系统中最常见的字体
plt.rcParams["font.family"] = ["SimHei", "Microsoft YaHei", "sans-serif"]
plt.rcParams["axes.unicode_minus"] = False

class Vanke1Strategy:
    """万科A迭代1策略：基础框架·估值+日线趋势（解决"回撤过大"）
    通过万科A自身估值定仓位、日线趋势过滤假信号，降低熊市回撤
    """
    
    def __init__(self, start_date, end_date, initial_capital=1000000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.stock_code = "sz.000002"  # 万科A代码
        self.stock_name = "万科A"
        self.etf_code = "sz.159001"  # 华宝添益货币基金，用于模拟现金仓位
        self.data = None
        self.portfolio = None
        self.signals = None
        self.positions = None
        self.transaction_cost = 0.0011  # 交易成本：佣金0.01%+滑点0.1%
        self.etf_annual_return = 0.02  # 货币基金年化收益率2%
        self.cash_floor = 0.1  # 现金仓下限：10%
        # 为了模拟，我们使用预设的PE分位数数据
        # 在实际应用中，应从理杏仁或同花顺获取真实的PE分位数数据
        self.pe_percentiles = None
        
    def get_data(self):
        """获取交易数据，从baostock获取万科A的真实数据"""
        # 尝试从baostock获取数据
        try:
            # 登录baostock
            lg = bs.login()
            print('login respond error_code:'+lg.error_code)
            print('login respond  error_msg:'+lg.error_msg)
            
            if lg.error_code != '0':
                print(f"登录baostock失败，错误码: {lg.error_code}，错误信息: {lg.error_msg}")
                return None
            
            # 获取交易数据
            rs = bs.query_history_k_data_plus(
                self.stock_code,
                "date,code,open,high,low,close,volume,amount,adjustflag",
                start_date=self.start_date,
                end_date=self.end_date,
                frequency="d",
                adjustflag="3"  # 复权类型，3为前复权
            )
            
            if rs.error_code != '0':
                print(f"获取日线数据失败: {rs.error_msg}")
                bs.logout()
                return None
            
            # 处理交易数据
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
            
            if len(data_list) == 0:
                print("警告: 未获取到任何K线数据")
                bs.logout()
                return None
                
            # 转换为DataFrame
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 将数值列转换为数值类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            print(f"成功获取 {self.stock_code}({self.stock_name}) 的日线数据，共 {len(df)} 条记录")
            
            # 登出baostock
            bs.logout()
            
            # 计算均线指标
            df['ma5'] = df['close'].rolling(window=5).mean()
            df['ma10'] = df['close'].rolling(window=10).mean()
            df['ma20'] = df['close'].rolling(window=20).mean()
            
            # 模拟PE分位数数据（在实际应用中应使用真实数据）
            # 这里为了演示，使用随机生成的数据，但遵循文档中提到的历史案例
            df['pe_percentile'] = np.random.rand(len(df)) * 100
            
            # 根据文档中的历史案例调整PE分位数
            # 2014年、2018年、2022年：低估（<30%）
            df.loc[(df.index.year == 2014) | (df.index.year == 2018) | (df.index.year == 2022), 'pe_percentile'] = np.random.rand(len(df[(df.index.year == 2014) | (df.index.year == 2018) | (df.index.year == 2022)])) * 30
            # 2019年、2023年：合理（30%-70%）
            df.loc[(df.index.year == 2019) | (df.index.year == 2023), 'pe_percentile'] = 30 + np.random.rand(len(df[(df.index.year == 2019) | (df.index.year == 2023)])) * 40
            # 2015年、2021年：高估（>70%）
            df.loc[(df.index.year == 2015) | (df.index.year == 2021), 'pe_percentile'] = 70 + np.random.rand(len(df[(df.index.year == 2015) | (df.index.year == 2021)])) * 30
            
            # 保存PE分位数数据
            self.pe_percentiles = df['pe_percentile']
            
            # 保存数据
            self.data = df
            
            # 打印最近10天的数据
            print("\n=== 最近10天的数据 ===")
            print(self.data.tail(10)[['close', 'ma5', 'ma10', 'ma20', 'pe_percentile']].to_string())
            
        except Exception as e:
            print(f"获取数据时发生错误: {e}")
            try:
                bs.logout()
            except:
                pass
            return None
        
        return self.data
        
    def generate_signals(self):
        """生成交易信号：基于PE分位数和日线趋势"""
        if self.data is None or self.data.empty:
            print("警告: 没有数据用于生成交易信号")
            self.signals = pd.DataFrame()
            return self.signals
            
        self.signals = pd.DataFrame(index=self.data.index)
        
        # 基于PE分位数确定目标仓位
        self.signals['target_position'] = 0.5  # 默认合理仓位50%
        self.signals.loc[self.data['pe_percentile'] < 30, 'target_position'] = 0.8  # 低估：80%
        self.signals.loc[self.data['pe_percentile'] > 70, 'target_position'] = 0.3  # 高估：30%
        
        # 判断日线趋势
        self.signals['bull_trend'] = (self.data['ma5'] > self.data['ma10']) & (self.data['ma10'] > self.data['ma20'])
        self.signals['bear_trend'] = (self.data['ma5'] < self.data['ma10']) & (self.data['ma10'] < self.data['ma20'])
        
        # 基于趋势调整仓位
        self.signals['adjusted_position'] = self.signals['target_position']
        self.signals.loc[self.signals['bear_trend'], 'adjusted_position'] = self.signals['target_position'] * 0.7  # 空头趋势：仓位下浮30%
        
        # 确保现金仓不低于10%
        self.signals['adjusted_position'] = np.minimum(self.signals['adjusted_position'], 1 - self.cash_floor)
        
        # 为了平滑仓位变化，使用移动平均
        self.signals['final_position'] = self.signals['adjusted_position'].rolling(window=5).mean().fillna(0.5)
        
        print(f"生成信号的数据条数: {len(self.data)}")
        print(f"仓位分布统计：\n{self.signals['final_position'].describe()}")
        
        return self.signals
        
    def backtest(self):
        """回测策略：基于PE分位数和日线趋势的仓位调整策略"""
        if self.signals is None or self.signals.empty:
            print("警告: 没有信号数据用于回测")
            self.portfolio = pd.DataFrame()
            return self.portfolio
            
        # 初始化仓位
        self.positions = pd.DataFrame(index=self.signals.index)
        self.positions[self.stock_code] = self.signals['final_position']
        self.positions[self.etf_code] = 1 - self.signals['final_position']  # 剩余资金买入货币基金
        
        # 初始化portfolio
        self.portfolio = pd.DataFrame(index=self.signals.index)
        self.portfolio['cash'] = self.initial_capital
        
        # 计算持仓价值
        if not self.data.empty and 'close' in self.data.columns and len(self.data['close']) > 0:
            # 计算股票持仓价值
            stock_investment = self.initial_capital * self.positions[self.stock_code].iloc[0] * (1 - self.transaction_cost)
            self.portfolio['stock_holdings'] = stock_investment * (self.data['close'] / self.data['close'].iloc[0])
            
            # 计算货币基金持仓价值（考虑2%的年化收益）
            etf_investment = self.initial_capital - stock_investment
            daily_return = (1 + self.etf_annual_return) ** (1/365) - 1
            cumulative_return = np.cumprod(1 + np.repeat(daily_return, len(self.data)))
            self.portfolio['etf_holdings'] = etf_investment * cumulative_return
            
            print(f"计算持仓价值成功，首日期价格: {self.data['close'].iloc[0]}")
        else:
            self.portfolio['stock_holdings'] = 0.0
            self.portfolio['etf_holdings'] = self.initial_capital
            print("警告: 无法计算持仓价值")
            
        # 计算调仓后的资产价值（从第二天开始）
        for i in range(1, len(self.portfolio)):
            # 计算上一天的总资产
            prev_total = self.portfolio['stock_holdings'].iloc[i-1] + self.portfolio['etf_holdings'].iloc[i-1]
            
            # 计算当天的目标仓位
            stock_weight = self.positions[self.stock_code].iloc[i]
            etf_weight = self.positions[self.etf_code].iloc[i]
            
            # 计算理想的持仓价值
            ideal_stock = prev_total * stock_weight
            ideal_etf = prev_total * etf_weight
            
            # 计算实际持仓价值（考虑交易成本）
            if ideal_stock > self.portfolio['stock_holdings'].iloc[i-1]:
                # 加仓
                self.portfolio['stock_holdings'].iloc[i] = ideal_stock * (1 - self.transaction_cost)
                self.portfolio['etf_holdings'].iloc[i] = prev_total - (ideal_stock * (1 - self.transaction_cost))
            elif ideal_stock < self.portfolio['stock_holdings'].iloc[i-1]:
                # 减仓
                self.portfolio['stock_holdings'].iloc[i] = ideal_stock
                self.portfolio['etf_holdings'].iloc[i] = (prev_total - ideal_stock) * (1 - self.transaction_cost)
            else:
                # 保持仓位
                self.portfolio['stock_holdings'].iloc[i] = self.portfolio['stock_holdings'].iloc[i-1] * (self.data['close'].iloc[i] / self.data['close'].iloc[i-1])
                self.portfolio['etf_holdings'].iloc[i] = self.portfolio['etf_holdings'].iloc[i-1] * (1 + daily_return)
        
        # 计算总资产和收益率
        self.portfolio['total'] = self.portfolio['stock_holdings'] + self.portfolio['etf_holdings']
        self.portfolio['returns'] = self.portfolio['total'].pct_change()
        
        # 计算基准收益（买入并持有）
        benchmark_investment = self.initial_capital * (1 - self.transaction_cost)
        self.portfolio['benchmark'] = benchmark_investment * (self.data['close'] / self.data['close'].iloc[0])
        
        # 调试输出portfolio信息
        print(f"portfolio数据形状: {self.portfolio.shape}")
        if not self.portfolio.empty:
            print(f"portfolio前5行:\n{self.portfolio[['stock_holdings', 'etf_holdings', 'total', 'benchmark']].head()}")
        
        return self.portfolio
        
    def analyze(self):
        """分析回测结果"""
        # 检查portfolio是否有有效数据
        if self.portfolio is None or self.portfolio.empty or 'total' not in self.portfolio.columns:
            print("警告: 没有足够的回测数据进行分析")
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'bench_total_return': 0.0,
                'bench_annual_return': 0.0
            }
            
        # 确保有足够的数据点
        if len(self.portfolio) < 2:
            print("警告: 数据点不足，无法进行完整分析")
            return {
                'total_return': 0.0,
                'annual_return': 0.0,
                'max_drawdown': 0.0,
                'bench_total_return': 0.0,
                'bench_annual_return': 0.0
            }
        
        # 计算策略表现指标
        try:
            total_return = (self.portfolio['total'].iloc[-1] / self.initial_capital - 1) * 100
            
            # 计算年化收益率
            n_years = (self.portfolio.index[-1] - self.portfolio.index[0]).days / 365.25
            annual_return = ((1 + total_return / 100) ** (1 / n_years)) - 1
            annual_return = annual_return * 100
            
            # 计算最大回撤
            rolling_max = self.portfolio['total'].cummax()
            daily_drawdown = self.portfolio['total'] / rolling_max - 1.0
            max_drawdown = daily_drawdown.min() * 100
            
            # 计算基准表现指标
            bench_total_return = (self.portfolio['benchmark'].iloc[-1] / self.initial_capital - 1) * 100
            bench_annual_return = ((1 + bench_total_return / 100) ** (1 / n_years)) - 1
            bench_annual_return = bench_annual_return * 100
            
            # 打印结果
            print(f"迭代1策略表现（{self.start_date}至{self.end_date}）：")
            print(f"期初资产: {self.initial_capital:.2f}元")
            print(f"期末资产: {self.portfolio['total'].iloc[-1]:.2f}元")
            print(f"总收益率: {total_return:.2f}%")
            print(f"年化收益率: {annual_return:.2f}%")
            print(f"最大回撤: {max_drawdown:.2f}%")
            print("\n基准（纯持有）表现：")
            print(f"期初资产: {self.initial_capital:.2f}元")
            print(f"期末资产: {self.portfolio['benchmark'].iloc[-1]:.2f}元")
            print(f"总收益率: {bench_total_return:.2f}%")
            print(f"年化收益率: {bench_annual_return:.2f}%")
            print("\n相对基准提升：")
            print(f"总收益率提升: {total_return - bench_total_return:.2f}个百分点")
            print(f"最大回撤降低: {max_drawdown - bench_max_drawdown:.2f}个百分点")
            
            # 计算基准的最大回撤用于比较
            bench_rolling_max = self.portfolio['benchmark'].cummax()
            bench_daily_drawdown = self.portfolio['benchmark'] / bench_rolling_max - 1.0
            bench_max_drawdown = bench_daily_drawdown.min() * 100
            
            # 绘制资产曲线
            plt.figure(figsize=(12, 8))
            plt.plot(self.portfolio['total'] / 10000, label='策略资产（万元）')
            plt.plot(self.portfolio['benchmark'] / 10000, label='基准资产（万元）')
            plt.title('万科A迭代1策略资产曲线')
            plt.xlabel('日期')
            plt.ylabel('资产（万元）')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # 绘制回撤曲线
            plt.figure(figsize=(12, 6))
            plt.plot(daily_drawdown * 100, label='策略回撤百分比（%）')
            plt.plot(bench_daily_drawdown * 100, label='基准回撤百分比（%）')
            plt.title('万科A迭代1策略回撤曲线')
            plt.xlabel('日期')
            plt.ylabel('回撤百分比（%）')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # 绘制仓位变化曲线
            plt.figure(figsize=(12, 6))
            plt.plot(self.signals['final_position'] * 100, label='股票仓位（%）')
            plt.title('万科A迭代1策略仓位变化')
            plt.xlabel('日期')
            plt.ylabel('仓位百分比（%）')
            plt.legend()
            plt.grid(True)
            plt.show()
            
        except Exception as e:
            print(f"分析过程中出错: {e}")
            total_return = 0.0
            annual_return = 0.0
            max_drawdown = 0.0
            bench_total_return = 0.0
            bench_annual_return = 0.0
            bench_max_drawdown = 0.0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'bench_total_return': bench_total_return,
            'bench_annual_return': bench_annual_return,
            'bench_max_drawdown': bench_max_drawdown
        }

if __name__ == "__main__":
    # 创建策略实例
    strategy = Vanke1Strategy(start_date="2014-10-01", end_date="2024-09-30")
    
    # 获取数据
    print("获取数据中...")
    data = strategy.get_data()
    
    if data is None or data.empty:
        print("无法获取数据，程序终止")
        exit()
    
    # 生成信号
    print("生成交易信号...")
    strategy.generate_signals()
    
    # 回测
    print("执行回测...")
    strategy.backtest()
    
    # 分析结果
    print("分析回测结果...")
    results = strategy.analyze()