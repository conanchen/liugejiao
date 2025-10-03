import baostock as bs
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

class Iteration1Strategy:
    """迭代1：估值+趋势双维度策略"""
    
    def __init__(self, start_date, end_date, initial_capital=1000000):
        self.start_date = start_date
        self.end_date = end_date
        self.initial_capital = initial_capital
        self.stock_code = "sz.000002"  # 万科A代码
        self.stock_name = "万科A"
        self.cash_etf_code = "511990"  # 华宝添益代码
        self.cash_etf_name = "华宝添益"
        self.data = None
        self.signals = None
        self.positions = None
        self.portfolio = None
        
    def get_data(self):
        """获取交易数据和PE数据，从baostock获取万科A的真实数据"""
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
                "date,code,open,high,low,close,volume,amount,adjustflag,peTTM",
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
            
            print(f"获取到的K线数据条数: {len(data_list)}")
            
            if len(data_list) == 0:
                print("警告: 未获取到任何K线数据")
                bs.logout()
                return None
                
            df = pd.DataFrame(data_list, columns=rs.fields)
            
            # 将数值列转换为数值类型
            numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount', 'peTTM']
            for col in numeric_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 转换日期列
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            print(f"成功获取 {self.stock_code}({self.stock_name}) 的日线数据，共 {len(df)} 条记录")
            
            # 处理PE数据
            if 'peTTM' in df.columns:
                # 数据清洗：将无效值替换为NaN
                df['peTTM'] = pd.to_numeric(df['peTTM'], errors='coerce')
                
                # 检查PE数据的有效性
                valid_pe_count = df['peTTM'].count()
                total_count = len(df)
                print(f"有效PE数据条数: {valid_pe_count}/{total_count}")
                
                # 填充缺失值，使用前向填充后反向填充
                df['peTTM'] = df['peTTM'].ffill().bfill()
            else:
                print("警告: 数据中不包含PE字段")
                bs.logout()
                return None
                
            # 合并数据
            self.data = df
            
            # 计算PE分位数（近10年滚动计算）
            window_size = 2520  # 10年约2520个交易日
            window_size = min(window_size, len(self.data) // 2)  # 取10年或数据一半的较小值
            print(f"使用的PE分位数计算窗口: {window_size}个交易日")
            
            # 计算PE分位数
            def calculate_quantile(x):
                if len(x) < 10:
                    return 0.5
                return x.rank(pct=True).iloc[-1]
            
            self.data['pe_quantile'] = self.data['peTTM'].rolling(window=window_size, min_periods=10).apply(calculate_quantile, raw=False)
            
            # 计算均线（5日、10日、20日）
            self.data['ma5'] = self.data['close'].rolling(window=5).mean()
            self.data['ma10'] = self.data['close'].rolling(window=10).mean()
            self.data['ma20'] = self.data['close'].rolling(window=20).mean()
            
            # 登出baostock
            bs.logout()
            
            # 打印最近10天的数据
            print("\n=== 最近10天的数据 ===")
            print(self.data.tail(10).to_string())
            
            # 保存K线数据到CSV文件
            csv_filename = f"iteration_1_{self.stock_code}.csv"
            self.data.to_csv(csv_filename, encoding="utf-8-sig")
            print(f"K线数据已保存到: {csv_filename}")
            
        except Exception as e:
            print(f"获取数据时发生错误: {e}")
            try:
                bs.logout()
            except:
                pass
            return None
        
        return self.data
        
    def generate_signals(self):
        """生成交易信号：严格按照文档中的核心策略规则实现"""
        if self.data is None or self.data.empty:
            print("警告: 没有数据用于生成交易信号")
            self.signals = pd.DataFrame()
            return self.signals
            
        self.signals = pd.DataFrame(index=self.data.index)
        
        # 调试输出信号数据信息
        print(f"生成信号的数据条数: {len(self.data)}")
        
        # 定义多头条件：5日线 > 10日线 > 20日线
        self.signals['bullish'] = (self.data['ma5'] > self.data['ma10']) & (self.data['ma10'] > self.data['ma20'])
        
        # 定义空头条件：5日线 < 10日线 < 20日线
        self.signals['bearish'] = (self.data['ma5'] < self.data['ma10']) & (self.data['ma10'] < self.data['ma20'])
        
        # 初始化信号为默认仓位0
        self.signals['signal'] = 0.0
        
        # 严格按照文档中的仓位规则设置：
        # 1. PE分位数<30%（低估）→80%仓位，但需满足多头排列才允许加仓
        self.signals.loc[(self.data['pe_quantile'] < 0.3) & self.signals['bullish'], 'signal'] = 0.8
        
        # 2. PE分位数30%-70%（合理）→50%仓位
        self.signals.loc[(self.data['pe_quantile'] >= 0.3) & (self.data['pe_quantile'] <= 0.7), 'signal'] = 0.5
        
        # 3. PE分位数>70%（高估）→30%仓位
        self.signals.loc[self.data['pe_quantile'] > 0.7, 'signal'] = 0.3
        
        # 4. 空头排列强制减至基础仓位的70%
        # 对每个PE区间，空头时仓位为基础仓位的70%
        self.signals.loc[(self.data['pe_quantile'] < 0.3) & self.signals['bearish'], 'signal'] = 0.8 * 0.7
        self.signals.loc[(self.data['pe_quantile'] >= 0.3) & (self.data['pe_quantile'] <= 0.7) & self.signals['bearish'], 'signal'] = 0.5 * 0.7
        self.signals.loc[(self.data['pe_quantile'] > 0.7) & self.signals['bearish'], 'signal'] = 0.3 * 0.7
        
        # 调试输出信号分布
        print(f"信号分布:\n{self.signals['signal'].value_counts()}")
        
        return self.signals
        
    def backtest(self):
        """回测策略：模拟资金管理实操表中的操作"""
        if self.signals is None or self.signals.empty:
            print("警告: 没有信号数据用于回测")
            self.positions = pd.DataFrame()
            self.portfolio = pd.DataFrame()
            return self.portfolio
            
        # 初始化仓位
        self.positions = pd.DataFrame(index=self.signals.index).fillna(0.0)
        self.positions[self.stock_code] = self.signals['signal']  # 持有股票的比例
        
        # 添加华宝添益仓位（闲置资金）
        self.positions[self.cash_etf_code] = 1.0 - self.signals['signal']
        
        # 调试输出仓位数据信息
        print(f"回测数据条数: {len(self.positions)}")
        
        # 初始化 portfolio
        self.portfolio = pd.DataFrame(index=self.signals.index).fillna(0.0)
        
        # 计算股票持仓价值
        if not self.data.empty and 'close' in self.data.columns and len(self.data['close']) > 0:
            # 股票持仓价值
            self.portfolio['stock_holdings'] = self.positions[self.stock_code] * self.initial_capital * (self.data['close'] / self.data['close'].iloc[0])
            
            # 华宝添益持仓价值（假设年化收益率2%，简化计算）
            # 实际应该使用真实的华宝添益数据，但这里为了简化采用近似计算
            daily_return = (1 + 0.02) ** (1/252) - 1
            cum_returns = (1 + daily_return) ** np.arange(len(self.portfolio))
            self.portfolio['cash_etf_holdings'] = self.positions[self.cash_etf_code] * self.initial_capital * cum_returns
            
            print(f"计算持仓价值成功，首日期价格: {self.data['close'].iloc[0]}")
        else:
            self.portfolio['stock_holdings'] = 0.0
            self.portfolio['cash_etf_holdings'] = 0.0
            print("警告: 无法计算持仓价值")
            
        # 计算总资产
        self.portfolio['total'] = self.portfolio['stock_holdings'] + self.portfolio['cash_etf_holdings']
        self.portfolio['returns'] = self.portfolio['total'].pct_change()
        
        # 计算基准收益（纯持有）
        if not self.data.empty and 'close' in self.data.columns and len(self.data['close']) > 0:
            self.portfolio['benchmark'] = self.initial_capital * (self.data['close'] / self.data['close'].iloc[0])
        else:
            self.portfolio['benchmark'] = self.initial_capital
            
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
            annual_return = ((1 + total_return / 100) **(252 / len(self.portfolio))) - 1
            annual_return = annual_return * 100
            
            # 计算最大回撤
            rolling_max = self.portfolio['total'].cummax()
            daily_drawdown = self.portfolio['total'] / rolling_max - 1.0
            max_drawdown = daily_drawdown.min() * 100
            
            # 计算基准表现指标
            if 'benchmark' in self.portfolio.columns and not self.portfolio['benchmark'].isna().all():
                bench_total_return = (self.portfolio['benchmark'].iloc[-1] / self.initial_capital - 1) * 100
                bench_annual_return = ((1 + bench_total_return / 100)** (252 / len(self.portfolio))) - 1
                bench_annual_return = bench_annual_return * 100
            else:
                bench_total_return = 0.0
                bench_annual_return = 0.0
            
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
            
            # 绘制资产曲线
            plt.figure(figsize=(12, 8))
            plt.plot(self.portfolio['total'] / 10000, label='策略资产（万元）')
            if 'benchmark' in self.portfolio.columns:
                plt.plot(self.portfolio['benchmark'] / 10000, label='基准资产（万元）')
            plt.title('迭代1策略与基准资产对比')
            plt.xlabel('日期')
            plt.ylabel('资产（万元）')
            plt.legend()
            plt.grid(True)
            plt.show()
            
            # 绘制仓位变化
            if self.positions is not None and self.stock_code in self.positions.columns:
                plt.figure(figsize=(12, 6))
                plt.plot(self.positions[self.stock_code] * 100, label='仓位比例（%）')
                plt.title('迭代1策略仓位变化')
                plt.xlabel('日期')
                plt.ylabel('仓位比例（%）')
                plt.ylim(0, 100)
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
    strategy = Iteration1Strategy(start_date="2014-10-01", end_date="2024-09-30")
    
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
