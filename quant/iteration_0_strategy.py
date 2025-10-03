import pandas as pd
import numpy as np
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

class Iteration0Strategy:
    """迭代0：买入并持有策略
    无任何交易规则，买入后长期持有
    """
    
    def __init__(self, start_date, end_date, initial_capital=1000000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.stock_code = "sz.000002"  # 万科A代码
        self.stock_name = "万科A"
        self.data = None
        self.portfolio = None
        self.signals = None
        self.positions = None
        
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
            
            # 保存数据
            self.data = df
            
            # 打印最近10天的数据
            print("\n=== 最近10天的数据 ===")
            print(self.data.tail(10).to_string())
            
        except Exception as e:
            print(f"获取数据时发生错误: {e}")
            try:
                bs.logout()
            except:
                pass
            return None
        
        return self.data
        
    def generate_signals(self):
        """生成交易信号：买入并持有策略只有一个买入信号"""
        if self.data is None or self.data.empty:
            print("警告: 没有数据用于生成交易信号")
            self.signals = pd.DataFrame()
            return self.signals
            
        self.signals = pd.DataFrame(index=self.data.index)
        self.signals['signal'] = 0.0
        
        # 在第一个交易日买入，之后一直持有
        self.signals.iloc[0, self.signals.columns.get_loc('signal')] = 1.0
        
        print(f"生成信号的数据条数: {len(self.data)}")
        print(f"信号分布: 买入信号在第1天")
        
        return self.signals
        
    def backtest(self):
        """回测策略：买入并持有"""
        if self.signals is None or self.signals.empty:
            print("警告: 没有信号数据用于回测")
            self.portfolio = pd.DataFrame()
            return self.portfolio
            
        # 初始化仓位：买入后100%持有
        self.positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        self.positions[self.stock_code] = 1.0  # 始终持有100%
        
        # 初始化portfolio
        self.portfolio = pd.DataFrame(index=self.signals.index).fillna(0.0)
        self.portfolio['cash'] = self.initial_capital - (self.positions[self.stock_code] * self.initial_capital)
        
        # 计算持仓价值
        if not self.data.empty and 'close' in self.data.columns and len(self.data['close']) > 0:
            self.portfolio['holdings'] = self.positions[self.stock_code] * self.initial_capital * (self.data['close'] / self.data['close'].iloc[0])
            print(f"计算持仓价值成功，首日期价格: {self.data['close'].iloc[0]}")
        else:
            self.portfolio['holdings'] = 0.0
            print("警告: 无法计算持仓价值")
            
        self.portfolio['total'] = self.portfolio['cash'] + self.portfolio['holdings']
        self.portfolio['returns'] = self.portfolio['total'].pct_change()
        
        # 计算基准收益（完全相同，因为是买入并持有策略）
        self.portfolio['benchmark'] = self.portfolio['total']
        
        # 调试输出portfolio信息
        print(f"portfolio数据形状: {self.portfolio.shape}")
        if not self.portfolio.empty:
            print(f"portfolio前5行:\n{self.portfolio.head()}")
        
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
            
            # 计算基准表现指标（与策略相同）
            bench_total_return = total_return
            bench_annual_return = annual_return
            
            # 打印结果
            print(f"迭代0策略表现（{self.start_date}至{self.end_date}）：")
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
            
            # 绘制资产曲线
            plt.figure(figsize=(12, 8))
            plt.plot(self.portfolio['total'] / 10000, label='策略资产（万元）')
            plt.title('迭代0策略资产曲线')
            plt.xlabel('日期')
            plt.ylabel('资产（万元）')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # 绘制回撤曲线
            plt.figure(figsize=(12, 6))
            plt.plot(daily_drawdown * 100, label='回撤百分比（%）')
            plt.title('迭代0策略回撤曲线')
            plt.xlabel('日期')
            plt.ylabel('回撤百分比（%）')
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
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'bench_total_return': bench_total_return,
            'bench_annual_return': bench_annual_return
        }

if __name__ == "__main__":
    # 创建策略实例
    strategy = Iteration0Strategy(start_date="2014-10-01", end_date="2024-09-30")
    
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