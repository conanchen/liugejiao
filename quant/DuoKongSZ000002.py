import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import baostock as bs
import os

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 用于存储markdown输出内容的变量
markdown_output = []

# 用于存储所有交易操作的列表（包括应该加仓、减仓的操作及结果）
all_trades_data = []

def add_markdown(text):
    """将文本添加到markdown输出中"""
    global markdown_output
    markdown_output.append(text)

def save_to_markdown(stock_code="sz.000002"):
    """将markdown内容保存到文件"""
    file_path = rf"d:\git\liugejiao\alltrends\DuoKong{stock_code}.md"
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(markdown_output))
    print(f"回测结果已保存到: {file_path}")

# 请根据多空定义及组合场景使用baostock获取近10年真实数据回测万科A，适时加减仓，得出各场景收益率及总收益率。

# 1. 初始化baostock并获取数据
def get_kline_data(start_date, end_date, stock_code="sz.000002", price_scale=1):
    lg = bs.login()
    if lg.error_code != '0':
        add_markdown(f"### 登录失败\n{lg.error_msg}")
        return None, None
    
    # 获取日线数据
    rs_daily = bs.query_history_k_data_plus(
        stock_code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 前复权
    )
    daily_df = rs_daily.get_data()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        daily_df[col] = pd.to_numeric(daily_df[col])
    
    # 将日线价格缩小指定倍数
    for col in ['open', 'high', 'low', 'close']:
        daily_df[col] = daily_df[col] / price_scale
    
    # 获取周线数据
    rs_weekly = bs.query_history_k_data_plus(
        stock_code,
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="w",
        adjustflag="2"  # 前复权
    )
    weekly_df = rs_weekly.get_data()
    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        weekly_df[col] = pd.to_numeric(weekly_df[col])
    
    # 将周线价格缩小指定倍数
    for col in ['open', 'high', 'low', 'close']:
        weekly_df[col] = weekly_df[col] / price_scale
    
    bs.logout()
    return daily_df, weekly_df

# 2. 计算技术指标
def calculate_indicators(df, is_weekly=True):
    df = df.copy()
    # 均线计算
    periods = [5, 10, 20, 60]
    for period in periods:
        df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    
    # 均线斜率计算(角度)
    for period in periods:
        df[f'ma{period}_slope'] = np.degrees(np.arctan(
            (df[f'ma{period}'] - df[f'ma{period}'].shift(10)) / 10  # 10周期斜率
        ))
    
    # MACD计算
    df['ema12'] = df['close'].ewm(span=12, adjust=False).mean()
    df['ema26'] = df['close'].ewm(span=26, adjust=False).mean()
    df['dif'] = df['ema12'] - df['ema26']
    df['dea'] = df['dif'].ewm(span=9, adjust=False).mean()
    df['macd_bar'] = (df['dif'] - df['dea']) * 2
    
    # 计算MACD Bar的移动
    df['macd_bar_prev'] = df['macd_bar'].shift(1)
    
    # K线涨跌占比
    df['up'] = df['close'] > df['open']
    df['up_ratio'] = df['up'].rolling(window=20).mean()  # 20周期内上涨占比
    
    return df

# 3. 判断多空场景 - 通用K线场景判断函数
def judge_kline_scene(kline_type, row):
    # 确定K线类型的中文前缀
    prefix_map = {
        'Weekly': '周线',
        'Daily': '日线',
        'M60': '60分钟线',
        'M30': '30分钟线',
        'M15': '15分钟线',
        'M5': '5分钟线',
        'M1': '1分钟线'
    }
    prefix = prefix_map.get(kline_type, kline_type)
    
    # 大多头判断(与周线大多头判断逻辑一致)
    # 1. 均线排列条件：5>10>20的均线排列
    all_ma_up = (row['ma5'] > row['ma10']) & (row['ma10'] > row['ma20'])
    
    # 2. 均线与60周期线的关系：5、10、20任意一条没有跌破60周期线
    no_ma_below_60 = (row['ma5'] > row['ma60']) | (row['ma10'] > row['ma60']) | (row['ma20'] > row['ma60'])
    
    # 3. 60周期线斜率条件：60周期线斜率向上
    ma60_slope_up = row['ma60_slope'] > 0  # 斜率大于0表示向上
    
    # 放宽判断条件，只要满足三个条件中的任意两个，即认为大多头场景存在
    if sum([all_ma_up, no_ma_below_60, ma60_slope_up]) >= 2:
        return f'{prefix}大多头'
    
    # 大空头判断(与大多头判断逻辑类似，条件相反)
    # 1. 均线排列条件：5<10<20的均线排列
    all_ma_down = (row['ma5'] < row['ma10']) & (row['ma10'] < row['ma20'])
    
    # 2. 均线与60周期线的关系：5、10、20任意一条没有突破60周期线
    no_ma_above_60 = (row['ma5'] < row['ma60']) | (row['ma10'] < row['ma60']) | (row['ma20'] < row['ma60'])
    
    # 3. 60周期线斜率条件：60周期线斜率向下
    ma60_slope_down = row['ma60_slope'] < 0  # 斜率小于0表示向下
    
    # 与大多头判断类似，只要满足三个条件中的任意两个，即认为大空头场景存在
    if sum([all_ma_down, no_ma_above_60, ma60_slope_down]) >= 2:
        return f'{prefix}大空头'
    
    return f'{prefix}震荡'

# 4. 组合场景判断
def get_combination_scene(weekly_scene, daily_scene):
    if weekly_scene == '周线大多头' and daily_scene == '日线大多头':
        return '多头太阳'
    elif weekly_scene == '周线大空头' and daily_scene == '日线大空头':
        return '空头太阴'
    elif weekly_scene == '周线大多头' and daily_scene == '日线大空头':
        return '多头小阴'
    elif weekly_scene == '周线大空头' and daily_scene == '日线大多头':
        return '空头小阳'
    else:
        return '震荡市'

# 5. 回测主函数
def backtest(stock_code="sz.000002"):
    # 获取近10年数据(2014-2024)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    daily_df, weekly_df = get_kline_data(start_date, end_date, stock_code=stock_code, price_scale=1)
    if daily_df is None:
        return
    
    # 计算指标
    daily_df = calculate_indicators(daily_df, is_weekly=False)
    weekly_df = calculate_indicators(weekly_df, is_weekly=True)
    
    # 判断场景 - 使用通用K线场景判断函数
    weekly_df['scene'] = weekly_df.apply(lambda row: judge_kline_scene('Weekly', row), axis=1)
    daily_df['scene'] = daily_df.apply(lambda row: judge_kline_scene('Daily', row), axis=1)
    
    # 合并周线场景到日线数据
    daily_df['weekly_scene'] = daily_df['date'].apply(
        lambda x: weekly_df[weekly_df['date'] <= x]['scene'].iloc[-1] 
        if not weekly_df[weekly_df['date'] <= x].empty else '周线震荡'
    )
    
    # 确定组合场景
    daily_df['combination_scene'] = daily_df.apply(
        lambda x: get_combination_scene(x['weekly_scene'], x['scene']), axis=1
    )
    
    # 初始化回测参数
    initial_capital = 1000000.0  # 100万总资金
    initial_position_value = 500000.0  # 期初50万仓位
    
    # 计算初始股数（假设使用第一天收盘价）
    first_close_price = daily_df['close'].iloc[0] if not daily_df.empty else 1.0
    initial_shares = initial_position_value / first_close_price
    
    daily_df['cash'] = initial_capital - initial_position_value  # 50万现金
    daily_df['shares'] = initial_shares
    daily_df['total_asset'] = initial_capital
    daily_df['position'] = initial_position_value / initial_capital  # 仓位比例 0.5
    
    # 记录交易事件
    buy_signals = []
    sell_signals = []
    trade_assets = []
    

    
    # 初始化加减仓计数器和历史记录
    max_trade_count = 3  # 最大加减仓次数限制
    buy_trade_count = 0  # 加仓次数计数
    sell_trade_count = 0  # 减仓次数计数
    last_buy_pos = daily_df['position'].iloc[0]  # 上次加仓后的仓位
    last_sell_pos = daily_df['position'].iloc[0]  # 上次减仓后的仓位
    
    # 记录前一日的日线场景和组合场景，用于检测场景变化
    prev_daily_scene = daily_df['scene'].iloc[0]
    prev_scene = daily_df['combination_scene'].iloc[0]

    # 添加调试：重点分析2019-2到2019-3期间的场景判断
    debug_period = (daily_df['date'] >= '2019-02-01') & (daily_df['date'] <= '2019-03-31')
    debug_df = daily_df[debug_period].copy()

    # 添加2019-2到2019-3期间分析到markdown
    add_markdown("## 2019-2到2019-3期间分析")
    add_markdown(f"- 总交易日数量: {len(debug_df)}")
    add_markdown(f"- 日线大多头天数: {len(debug_df[debug_df['scene'] == '日线大多头'])}")
    add_markdown(f"- 周线大多头天数: {len(debug_df[debug_df['weekly_scene'] == '周线大多头'])}")
    add_markdown(f"- 多头太阳场景(周多+日多)天数: {len(debug_df[debug_df['combination_scene'] == '多头太阳'])}")
    add_markdown(f"- 震荡市场景天数: {len(debug_df[debug_df['combination_scene'] == '震荡市'])}")

    # 显示日线大多头但周线不是大多头的情况
    daily_bull_weekly_not = debug_df[(debug_df['scene'] == '日线大多头') & (debug_df['weekly_scene'] != '周线大多头')]
    add_markdown(f"\n### 日线大多头但周线非大多头的天数: {len(daily_bull_weekly_not)}")
    if len(daily_bull_weekly_not) > 0:
        add_markdown("这些日期的周线场景都是: '周线震荡'")

    # 分析为何没有加仓
    add_markdown("\n### 策略仓位设置逻辑")
    add_markdown("- 只有当'周线大多头' + '日线大多头' = '多头太阳'时，才会设置90%的高仓位")
    add_markdown("- 当'周线震荡' + '日线大多头'时，仍归为'震荡市'，保持55%的中性仓位")
    add_markdown("- 因此2019-2到2019-3期间虽然日线出现大多头，但周线未确认，所以没有触发加仓")
    
    # 执行交易策略
    for i in range(1, len(daily_df)):
        prev = daily_df.iloc[i-1]
        current = daily_df.iloc[i]
        scene = current['combination_scene']
        current_daily_scene = current['scene']
        
        # 检测日线大空头结束，重置加仓最大次数
        if prev_daily_scene == '日线大空头' and current_daily_scene != '日线大空头':
            # 日线大空头结束，重置加仓次数，准备迎接可能的多头行情
            buy_trade_count = 0
        
        # 计算可交易资金和股份
        cash = prev['cash']
        shares = prev['shares']
        close_price = current['close']
        total_asset = cash + shares * close_price
        current_position = shares * close_price / total_asset
        
        # 初始化交易差异和目标股数为0
        diff = 0
        target_shares = shares  # 默认目标股数为当前持股数
        
        # 定义各场景的目标仓位
        if scene == '多头太阳':
            target_pos = 0.90  # 90% 高仓位
        elif scene == '空头太阴' or current_daily_scene == '日线大空头':
            target_pos = 0.10  # 10% 低仓位（日线大空头时的防御仓位）
        elif scene == '多头小阴':
            target_pos = 0.30  # 30% 较低仓位（多头小阴时降低仓位）
        elif scene == '空头小阳':
            target_pos = 0.40  # 40% 中等仓位（空头小阳时可以适度增加仓位）
        else:  # 震荡市
            target_pos = 0.55  # 55% 中性仓位
            
        # 确保震荡市场景不会突然变为10%的仓位，除非明确进入空头场景
        if scene == '震荡市' and prev_scene != '空头太阴' and prev_scene != '日线大空头':
            target_pos = 0.55  # 强制震荡市保持55%的中性仓位
        
        # 基于趋势变化和场景类型来触发交易
        # 更新前一日场景
        if i > 0:
            prev_scene = daily_df.iloc[i-1]['combination_scene']
            prev_daily_scene = daily_df.iloc[i-1]['scene']
            
            # 定义场景交易参数配置
            scene_params = {
                '多头太阳': {'action': 'buy', 'ratio': 0.20, 'step_factor_base': 0.15},
                '多头小阴': {'action': 'sell', 'ratio': 0.20, 'step_factor_base': 0.3},
                '空头太阴': {'action': 'sell', 'ratio': 0.20, 'step_factor_base': 0.15},
                '空头小阳': {'action': 'buy', 'ratio': 0.15, 'step_factor_base': 0.3}
            }
            
            # 检查是否为趋势反转
            is_reversal = ((prev_scene in ['多头太阳', '多头小阴'] and scene in ['空头太阴', '空头小阳']) or \
                         (prev_scene in ['空头太阴', '空头小阳'] and scene in ['多头太阳', '多头小阴']))
            
            # 处理趋势反转
            if is_reversal:
                target_shares = total_asset * target_pos / close_price
                diff = target_shares - shares
            
            # 处理场景变化时的交易
            elif scene in scene_params and prev_scene != scene:
                params = scene_params[scene]
                
                # 根据操作类型和交易次数限制执行交易
                if params['action'] == 'buy' and buy_trade_count < max_trade_count:
                    target_shares = total_asset * (current_position + params['ratio']) / close_price
                    diff = target_shares - shares
                    step_factor = 1.0 - (buy_trade_count * params['step_factor_base'])
                    diff = max(1, min(diff, diff * step_factor))
                elif params['action'] == 'sell' and sell_trade_count < max_trade_count:
                    target_shares = total_asset * (current_position - params['ratio']) / close_price
                    diff = target_shares - shares
                    step_factor = 1.0 - (sell_trade_count * params['step_factor_base'])
                    diff = min(-1, max(diff, diff * step_factor))
            
            # 处理震荡市场景的高抛低吸
            elif scene == '震荡市' and prev_scene == '震荡市' and i > 2:
                prev_close = daily_df.iloc[i-1]['close']
                sell_threshold = 0.02  # 上涨2%以上考虑减仓
                buy_threshold = -0.02  # 下跌2%以上考虑加仓
                price_change_pct = (close_price - prev_close) / prev_close
                
                # 高抛：价格上涨超过阈值且当前仓位高于目标仓位
                if price_change_pct > sell_threshold and current_position > target_pos and sell_trade_count < max_trade_count:
                    swing_sell_ratio = 0.05  # 减仓5%的总资产
                    target_shares = total_asset * (current_position - swing_sell_ratio) / close_price
                    diff = target_shares - shares
                    diff = min(-1, max(diff, diff * 0.9))  # 略微保守的调整
                # 低吸：价格下跌超过阈值且当前仓位低于目标仓位
                elif price_change_pct < buy_threshold and current_position < target_pos and buy_trade_count < max_trade_count:
                    swing_buy_ratio = 0.05  # 加仓5%的总资产
                    target_shares = total_asset * (current_position + swing_buy_ratio) / close_price
                    diff = target_shares - shares
                    diff = max(1, min(diff, diff * 0.9))  # 略微保守的调整
        
        # 确保交易股数至少为1股或-1股
        if abs(diff) > 0 and abs(diff) <= 1:
            diff = 0  # 交易数量太小，不执行交易
        elif diff > 0:
            diff = max(1, diff)  # 确保至少加仓1股
        elif diff < 0:
            diff = min(-1, diff)  # 确保至少减仓1股
        
        # 获取当前日期
        current_date = current['date']
        
        # 记录所有应该加仓、减仓的操作
        trade_status = "未执行"  # 默认状态
        trade_reason = ""  # 交易原因
        final_diff = diff  # 最终的交易量
        
        if diff > 0 and abs(diff) > 1:  # 应该加仓
            # 检查资金是否足够
            need_cash = diff * close_price
            if need_cash > cash:
                trade_status = "未执行"
                trade_reason = "资金不足"
            else:
                trade_status = "已执行" if not (buy_trade_count >= max_trade_count) else "未执行"
                if buy_trade_count >= max_trade_count:
                    trade_reason = "已达到最大加仓次数"
        elif diff < 0 and abs(diff) > 1:  # 应该减仓
            # 检查股份是否足够
            if shares + diff < 0:
                trade_status = "未执行"
                trade_reason = "持股不足"
            else:
                trade_status = "已执行" if not (sell_trade_count >= max_trade_count) else "未执行"
                if sell_trade_count >= max_trade_count:
                    trade_reason = "已达到最大减仓次数"
        else:
            if abs(diff) <= 1:
                trade_reason = "交易数量太小（≤1股）"
            
        # 添加到所有交易操作列表
        all_trades_data.append({
            'date': current_date.strftime('%Y-%m-%d'),
            'scene': scene,
            'total_asset': f"{total_asset:.2f}",
            'cash': f"{cash:.2f}",
            'shares': f"{shares:.2f}",
            'close_price': f"{close_price:.2f}",
            'target_pos': f"{target_pos:.2%}",
            'target_shares': f"{target_shares:.2f}",
            'diff': f"{diff:.2f}",
            'final_diff': f"{final_diff:.2f}",
            'trade_type': "加仓" if diff > 0 else "减仓" if diff < 0 else "无交易",
            'trade_status': trade_status,
            'reason': trade_reason
        })
        

        
        # 记录交易信号和对应资产值，包括趋势类型
        if diff > 0 and abs(diff) > 1:  # 加仓，且差异大于1股
            # 确定当前趋势类型
            trend_type = ''
            if scene in ['多头太阳', '多头小阴']:
                trend_type = '多头趋势'
            elif scene in ['空头太阴', '空头小阳']:
                trend_type = '空头趋势'
            else:
                trend_type = '振荡期'
            buy_signals.append((current['date'], current['close'], trend_type))
            # 计算交易后的预期资产值
            expected_asset = cash - diff * close_price + (shares + diff) * close_price
            trade_assets.append(expected_asset)
        elif diff < 0 and abs(diff) > 1:  # 减仓，且差异大于1股
            # 确定当前趋势类型
            trend_type = ''
            if scene in ['多头太阳', '多头小阴']:
                trend_type = '多头趋势'
            elif scene in ['空头太阴', '空头小阳']:
                trend_type = '空头趋势'
            else:
                trend_type = '振荡期'
            sell_signals.append((current['date'], current['close'], trend_type))
            # 计算交易后的预期资产值
            expected_asset = cash - diff * close_price + (shares + diff) * close_price
            trade_assets.append(expected_asset)
            
        # 更新前一日的日线场景
        prev_daily_scene = current_daily_scene
        
        if diff > 0:  # 加仓
            # 决定加仓时，重置减仓次数
            sell_trade_count = 0
            transaction_fee_rate = 0.0003  # 交易费率
            need_cash = diff * close_price * (1 + transaction_fee_rate)
            if need_cash <= cash:
                shares += diff
                cash -= need_cash
                # 更新加仓计数器和历史记录
                if buy_trade_count < max_trade_count:
                    buy_trade_count += 1
                    last_buy_pos = target_pos
            else:
                # 资金不足，调整购买数量
                max_possible_shares = int(cash / (close_price * (1 + transaction_fee_rate)))
                if max_possible_shares > 0:
                    actual_diff = max_possible_shares
                    actual_cash_needed = actual_diff * close_price * (1 + transaction_fee_rate)
                    shares += actual_diff
                    cash -= actual_cash_needed
                    # 更新加仓计数器和历史记录
                    if buy_trade_count < max_trade_count:
                        buy_trade_count += 1
                        last_buy_pos = target_pos
        elif diff < 0:  # 减仓
            # 决定减仓时，重置加仓次数
            buy_trade_count = 0
            transaction_fee_rate = 0.0003  # 交易费率
            # 计算实际可卖出的股数（确保不卖出过多）
            actual_diff = max(diff, -shares)  # 不能卖出超过持有数量
            if actual_diff < 0:
                cash_received = abs(actual_diff) * close_price * (1 - transaction_fee_rate)
                shares += actual_diff
                cash += cash_received
                # 更新减仓计数器和历史记录
                if sell_trade_count < max_trade_count:
                    sell_trade_count += 1
                    last_sell_pos = target_pos
        
        # 更新资产
        daily_df.at[i, 'cash'] = cash
        daily_df.at[i, 'shares'] = shares
        daily_df.at[i, 'total_asset'] = cash + shares * close_price
        daily_df.at[i, 'position'] = target_pos
        
    # 计算各场景收益率
    scenes = ['多头太阳', '空头太阴', '多头小阴', '空头小阳', '震荡市']
    scene_returns = {}
    scene_time_periods = {scene: [] for scene in scenes}  # 存储各场景的时间区间
    for scene in scenes:
        mask = daily_df['combination_scene'] == scene
        if mask.sum() < 2:
            scene_returns[scene] = 0.0
            continue
        start_asset = daily_df[mask].iloc[0]['total_asset']
        end_asset = daily_df[mask].iloc[-1]['total_asset']
        scene_returns[scene] = (end_asset / start_asset - 1) * 100
        
        # 收集该场景的所有连续时间段
        if mask.sum() > 0:
            # 找到所有连续的区间
            current_period_start = None
            for i in range(len(daily_df)):
                if mask.iloc[i]:
                    if current_period_start is None:
                        current_period_start = daily_df.iloc[i]['date']
                elif current_period_start is not None:
                    current_period_end = daily_df.iloc[i-1]['date']
                    scene_time_periods[scene].append((current_period_start, current_period_end))
                    current_period_start = None
            
            # 处理最后一个区间
            if current_period_start is not None:
                current_period_end = daily_df.iloc[-1]['date']
                scene_time_periods[scene].append((current_period_start, current_period_end))
    
    # 总收益率
    total_return = (daily_df.iloc[-1]['total_asset'] / initial_capital - 1) * 100
    
    # 添加所有交易操作详情表格到markdown（包含强制交易和普通交易）
    if all_trades_data:
        add_markdown("\n## 所有交易操作详情")
        # 添加表格表头
        add_markdown("| 日期 | 场景 | 当前总资产 | 当前现金 | 当前持有股数 | 当前价格 | 目标仓位 | 计算目标股数 | 建议交易股数 | 实际交易股数 | 交易类型 | 交易状态 | 原因/说明 |")
        add_markdown("| ---- | ---- | ---------- | -------- | ------------ | -------- | -------- | ------------ | ------------ | ------------ | -------- | -------- | ---------- |")
        # 添加表格数据
        for trade in all_trades_data:
            if trade['trade_type'] != "无交易":  # 只显示有交易建议的记录
                add_markdown(f"| {trade['date']} | {trade['scene']} | {trade['total_asset']} | {trade['cash']} | {trade['shares']} | {trade['close_price']} | {trade['target_pos']} | {trade['target_shares']} | {trade['diff']} | {trade['final_diff']} | {trade['trade_type']} | {trade['trade_status']} | {trade['reason']} |")
    
    # 添加加减仓逻辑说明到markdown
    add_markdown("\n## 加减仓逻辑说明")
    add_markdown("\n策略采用阶梯递减的加减仓机制，具体规则如下：\n")
    
    add_markdown("### 加仓逻辑\n")
    add_markdown("- 最大加仓次数限制为3次\n")
    add_markdown("- 采用阶梯递减模式：每次加仓量递减30%（递减因子分别为0.7, 0.4, 0.1）\n")
    add_markdown("- 加仓量计算公式：调整差异 = 目标差异 * 阶梯因子\n")
    add_markdown("- 每次加仓确保至少购买1股\n")
    add_markdown("- 当日线大空头结束后，会重置加仓次数计数器\n")
    
    add_markdown("### 减仓逻辑\n")
    add_markdown("- 普通场景：最大减仓次数限制为3次，每次减仓量递减30%（递减因子分别为0.7, 0.4, 0.1）\n")
    add_markdown("- 日线大空头场景：特殊处理，减仓速度加快，每次减仓量递减20%（递减因子分别为0.8, 0.6, 0.4）\n")
    add_markdown("- 减仓量计算公式：调整差异 = 目标差异 * 阶梯因子\n")
    add_markdown("- 每次减仓确保至少卖出1股\n")
    add_markdown("- 日线大空头场景下，目标仓位调整为10%，加速资金保护\n")
    
    add_markdown("### 交易执行条件\n")
    add_markdown("- 交易差异绝对值必须大于1股才会触发交易建议\n")
    add_markdown("- 加仓时会检查资金是否足够，资金不足时交易不会执行\n")
    add_markdown("- 超过最大交易次数后，不再执行加减仓操作\n")

    # 添加结果到markdown
    add_markdown("\n## 回测结果")
    add_markdown("### 各场景收益率")
    for scene, ret in scene_returns.items():
        add_markdown(f"- {scene}: {ret:.2f}%")
    
    # 添加各场景时间区间信息
    add_markdown("\n### 各场景时间区间")
    for scene in scenes:
        periods = scene_time_periods[scene]
        if not periods:
            add_markdown(f"- **{scene}**: 无数据")
        else:
            add_markdown(f"- **{scene}** ({len(periods)}个时间段):")
            for start, end in periods:
                # 格式化日期显示
                start_str = start.strftime('%Y-%m-%d') if hasattr(start, 'strftime') else str(start)
                end_str = end.strftime('%Y-%m-%d') if hasattr(end, 'strftime') else str(end)
                add_markdown(f"  - {start_str} 至 {end_str}")
    
    add_markdown(f"\n### 总收益率\n- {total_return:.2f}%")
    
    # 保存markdown文件
    save_to_markdown(stock_code=stock_code)
    
    # 可视化资产曲线
    plt.figure(figsize=(14, 8))  # 增大图表尺寸
    plt.plot(daily_df['date'], daily_df['total_asset'], label='策略资产', linewidth=2)
    
    # 绘制万科A股价曲线和均线
    # 对股价进行标准化，使其与策略资产在同一量级上显示
    normalized_price = initial_capital * daily_df['close'] / daily_df['close'].iloc[0]
    plt.plot(daily_df['date'], normalized_price, label='万科A', linewidth=2)
    
    # 计算并添加万科A股价的均线
    # 先对原始股价进行标准化，然后计算均线
    ma5_normalized = initial_capital * daily_df['ma5'] / daily_df['close'].iloc[0]
    ma10_normalized = initial_capital * daily_df['ma10'] / daily_df['close'].iloc[0]
    ma20_normalized = initial_capital * daily_df['ma20'] / daily_df['close'].iloc[0]
    ma60_normalized = initial_capital * daily_df['ma60'] / daily_df['close'].iloc[0]
    
    # 简化均线标签名称
    plt.plot(daily_df['date'], ma5_normalized, ':', label='5日均线', alpha=0.7, color='orange')
    plt.plot(daily_df['date'], ma10_normalized, ':', label='10日均线', alpha=0.7, color='purple')
    plt.plot(daily_df['date'], ma20_normalized, ':', label='20日均线', alpha=0.7, color='cyan')
    plt.plot(daily_df['date'], ma60_normalized, ':', label='60日均线', alpha=0.7, color='magenta')
    
    # 在万科A股价上添加加、减仓标记，按趋势类型区分
    if buy_signals:
        # 按趋势类型分组
        bull_buy_dates, bull_buy_prices = [], []
        bear_buy_dates, bear_buy_prices = [], []
        range_buy_dates, range_buy_prices = [], []
        
        for date, price, trend in buy_signals:
            if trend == '多头趋势':
                bull_buy_dates.append(date)
                bull_buy_prices.append(price)
            elif trend == '空头趋势':
                bear_buy_dates.append(date)
                bear_buy_prices.append(price)
            else:
                range_buy_dates.append(date)
                range_buy_prices.append(price)
        
        # 绘制不同趋势下的加仓标记
        if bull_buy_dates:
            bull_norm_prices = [initial_capital * p / daily_df['close'].iloc[0] for p in bull_buy_prices]
            plt.scatter(bull_buy_dates, bull_norm_prices, color='red', marker='^', label='多头趋势加仓', alpha=0.8)
        if bear_buy_dates:
            bear_norm_prices = [initial_capital * p / daily_df['close'].iloc[0] for p in bear_buy_prices]
            plt.scatter(bear_buy_dates, bear_norm_prices, color='blue', marker='^', label='空头趋势加仓', alpha=0.8)
        if range_buy_dates:
            range_norm_prices = [initial_capital * p / daily_df['close'].iloc[0] for p in range_buy_prices]
            plt.scatter(range_buy_dates, range_norm_prices, color='purple', marker='^', label='振荡期加仓', alpha=0.8)
    
    if sell_signals:
        # 按趋势类型分组
        bull_sell_dates, bull_sell_prices = [], []
        bear_sell_dates, bear_sell_prices = [], []
        range_sell_dates, range_sell_prices = [], []
        
        for date, price, trend in sell_signals:
            if trend == '多头趋势':
                bull_sell_dates.append(date)
                bull_sell_prices.append(price)
            elif trend == '空头趋势':
                bear_sell_dates.append(date)
                bear_sell_prices.append(price)
            else:
                range_sell_dates.append(date)
                range_sell_prices.append(price)
        
        # 绘制不同趋势下的减仓标记
        if bull_sell_dates:
            bull_norm_prices = [initial_capital * p / daily_df['close'].iloc[0] for p in bull_sell_prices]
            plt.scatter(bull_sell_dates, bull_norm_prices, color='green', marker='v', label='多头趋势减仓', alpha=0.8)
        if bear_sell_dates:
            bear_norm_prices = [initial_capital * p / daily_df['close'].iloc[0] for p in bear_sell_prices]
            plt.scatter(bear_sell_dates, bear_norm_prices, color='cyan', marker='v', label='空头趋势减仓', alpha=0.8)
        if range_sell_dates:
            range_norm_prices = [initial_capital * p / daily_df['close'].iloc[0] for p in range_sell_prices]
            plt.scatter(range_sell_dates, range_norm_prices, color='orange', marker='v', label='振荡期减仓', alpha=0.8)
    
    plt.title('回测资产曲线')
    plt.xlabel('日期')
    plt.ylabel('资产(元)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    backtest()