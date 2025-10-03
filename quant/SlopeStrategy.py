# @brief: 斜率策略，参照文件[SlopeDistribution_sz.000002.md]
# 采用baostock api 获取股票最近20年真实数据，计算斜率，支持指定股票代码，
# 根据市场9种组合场景，设计斜率策略，每个组合场景下，根据斜率分布设置目标仓位、加减仓逻辑
# 回测策略的分场景收益率、总收率，报告写入如SlopeStrategy_sz.000002.md，
# 由于要根据移动平均线M5、M10、M20、M60周/日来判断买入卖出信息，所以M60周计算到了才开始进行交易
# 上面的提示词不要删除

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

# 用于存储所有交易操作的列表
all_trades = []

# 场景配置字典，为每个场景设置不同的策略参数
scene_configs = {
    '多头太阳': {
        'target_position': 0.9,  # 目标仓位90%
        'buy_threshold': 0.85,   # 低于这个仓位时买入
        'sell_threshold': 0.95,  # 高于这个仓位时卖出
        'step_size': 0.05,       # 每次交易的仓位调整步长
        'description': '强烈上涨趋势，双周期共振向上，重仓持有',
        'golden_cross_increase_rate': 0.15,  # 多头场景下金叉阶梯加仓比例
        'death_cross_decrease_rate': 0.10    # 多头场景下死叉阶梯减仓比例
    },
    '多头少阳': {
        'target_position': 0.7,  # 目标仓位70%
        'buy_threshold': 0.65,
        'sell_threshold': 0.75,
        'step_size': 0.05,
        'description': '周线趋势向上，日线整理，中仓持有',
        'golden_cross_increase_rate': 0.12,
        'death_cross_decrease_rate': 0.08
    },
    '多头少阴': {
        'target_position': 0.5,  # 目标仓位50%
        'buy_threshold': 0.45,
        'sell_threshold': 0.55,
        'step_size': 0.05,
        'description': '周线趋势向上，日线短期回调，轻仓持有，回调结束后加仓',
        'golden_cross_increase_rate': 0.10,
        'death_cross_decrease_rate': 0.06
    },
    '震荡小阳': {
        'target_position': 0.6,  # 目标仓位60%
        'buy_threshold': 0.55,
        'sell_threshold': 0.65,
        'step_size': 0.05,
        'description': '周线整理，日线短期上涨，逢低买入',
        'golden_cross_increase_rate': 0.08,
        'death_cross_decrease_rate': 0.08
    },
    '震荡平衡': {
        'target_position': 0.4,  # 目标仓位40%
        'buy_threshold': 0.35,
        'sell_threshold': 0.45,
        'step_size': 0.05,
        'description': '双周期均处于整理状态，轻仓观望，高抛低吸',
        'golden_cross_increase_rate': 0.06,
        'death_cross_decrease_rate': 0.06
    },
    '震荡小阴': {
        'target_position': 0.3,  # 目标仓位30%
        'buy_threshold': 0.25,
        'sell_threshold': 0.35,
        'step_size': 0.05,
        'description': '周线整理，日线短期下跌，逢高卖出',
        'golden_cross_increase_rate': 0.05,
        'death_cross_decrease_rate': 0.09
    },
    '空头小阳': {
        'target_position': 0.2,  # 目标仓位20%
        'buy_threshold': 0.15,
        'sell_threshold': 0.25,
        'step_size': 0.05,
        'description': '周线趋势向下，日线短期反弹，轻仓参与，快进快出',
        'golden_cross_increase_rate': 0.04,
        'death_cross_decrease_rate': 0.12
    },
    '空头少阴': {
        'target_position': 0.1,  # 目标仓位10%
        'buy_threshold': 0.05,
        'sell_threshold': 0.15,
        'step_size': 0.05,
        'description': '周线趋势向下，日线整理，极低仓位，观望为主',
        'golden_cross_increase_rate': 0.03,
        'death_cross_decrease_rate': 0.15
    },
    '空头太阴': {
        'target_position': 0.0,  # 目标仓位0%
        'buy_threshold': 0.0,
        'sell_threshold': 0.0,
        'step_size': 0.0,
        'description': '强烈下跌趋势，双周期共振向下，空仓等待',
        'golden_cross_increase_rate': 0.0,
        'death_cross_decrease_rate': 0.20
    }
}

def add_markdown(text):
    """将文本添加到markdown输出中"""
    global markdown_output
    markdown_output.append(text)

def save_to_markdown(stock_code="sz.000002"):
    """将markdown内容保存到文件"""
    file_path = rf"d:\git\liugejiao\alltrends\SlopeStrategy_{stock_code}.md"
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(markdown_output))
    print(f"斜率策略回测报告已保存到: {file_path}")

def get_kline_data(start_date, end_date, stock_code="sz.000002"):
    """
    从baostock获取股票的日线和周线数据
    """
    print("登录baostock...")
    lg = bs.login()
    if lg.error_code != '0':
        print(f"登录失败: {lg.error_msg}")
        return None, None
    print("登录成功!")
    
    # 获取日线数据
    print("获取日线数据...")
    rs_daily = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="2"  # 前复权
    )
    daily_df = rs_daily.get_data()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        daily_df[col] = pd.to_numeric(daily_df[col])
    
    # 获取周线数据
    print("获取周线数据...")
    rs_weekly = bs.query_history_k_data_plus(
        stock_code,
        "date,code,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="w",
        adjustflag="2"  # 前复权
    )
    weekly_df = rs_weekly.get_data()
    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        weekly_df[col] = pd.to_numeric(weekly_df[col])
    
    bs.logout()
    print("登出成功!")
    return daily_df, weekly_df

def calculate_slope(df, window=10, period=60):
    """
    计算指定周期均线的斜率分布和M5/M10均线
    window: 计算斜率的窗口大小
    period: 均线周期
    """
    # 计算60日均线
    df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    
    # 计算均线斜率(角度)
    df[f'ma{period}_slope'] = np.degrees(np.arctan(
        (df[f'ma{period}'] - df[f'ma{period}'].shift(window)) / window  # window周期斜率
    ))
    
    # 计算M5和M10均线
    df['ma5'] = df['close'].rolling(window=5).mean()
    df['ma10'] = df['close'].rolling(window=10).mean()
    
    # 计算M5和M10的交叉状态
    df['ma5_gt_ma10'] = df['ma5'] > df['ma10']
    df['ma5_lt_ma10'] = df['ma5'] < df['ma10']
    
    # 判断金叉死叉状态
    df['golden_cross'] = False
    df['death_cross'] = False
    
    # 找到金叉点（M5由下而上穿越M10）
    golden_cross_mask = (df['ma5_gt_ma10'] & ~df['ma5_gt_ma10'].shift(1, fill_value=False))
    df.loc[golden_cross_mask, 'golden_cross'] = True
    
    # 找到死叉点（M5由上而下穿越M10）
    death_cross_mask = (df['ma5_lt_ma10'] & ~df['ma5_lt_ma10'].shift(1, fill_value=False))
    df.loc[death_cross_mask, 'death_cross'] = True
    
    # 计算M5在M10上方持续的天数
    df['ma5_above_ma10_days'] = 0
    consecutive_days = 0
    for i in range(len(df)):
        if df.loc[i, 'ma5_gt_ma10']:
            consecutive_days += 1
        else:
            consecutive_days = 0
        df.loc[i, 'ma5_above_ma10_days'] = consecutive_days
    
    # 计算M5在M10下方持续的天数
    df['ma5_below_ma10_days'] = 0
    consecutive_days = 0
    for i in range(len(df)):
        if df.loc[i, 'ma5_lt_ma10']:
            consecutive_days += 1
        else:
            consecutive_days = 0
        df.loc[i, 'ma5_below_ma10_days'] = consecutive_days
    
    return df

def judge_market_scene_by_slope(slope_value):
    """
    根据斜率值判断市场场景
    """
    if slope_value > 1:
        return '多头'
    elif slope_value < -1:
        return '空头'
    else:
        return '震荡'

def get_combination_scene(weekly_scene, daily_scene):
    """
    根据周线和日线场景判断组合场景
    完整的9种组合情况
    """
    if weekly_scene == '周线多头' and daily_scene == '日线多头':
        return '多头太阳'
    elif weekly_scene == '周线多头' and daily_scene == '日线震荡':
        return '多头少阳'
    elif weekly_scene == '周线多头' and daily_scene == '日线空头':
        return '多头少阴'
    elif weekly_scene == '周线震荡' and daily_scene == '日线多头':
        return '震荡小阳'
    elif weekly_scene == '周线震荡' and daily_scene == '日线震荡':
        return '震荡平衡'
    elif weekly_scene == '周线震荡' and daily_scene == '日线空头':
        return '震荡小阴'
    elif weekly_scene == '周线空头' and daily_scene == '日线多头':
        return '空头小阳'
    elif weekly_scene == '周线空头' and daily_scene == '日线震荡':
        return '空头少阴'
    elif weekly_scene == '周线空头' and daily_scene == '日线空头':
        return '空头太阴'
    else:
        return '未知场景'

def backtest_strategy(daily_df, weekly_df, initial_capital=1000000.0):
    """
    根据9种组合场景和M5/M10均线交叉进行策略回测
    """
    # 为日线数据添加周线场景信息
    daily_df['weekly_ma60_slope'] = daily_df['date'].apply(
        lambda x: weekly_df[weekly_df['date'] <= x]['ma60_slope'].iloc[-1] 
        if not weekly_df[weekly_df['date'] <= x].empty else 0
    )
    
    # 判断每日的市场场景
    daily_df['daily_scene'] = daily_df['ma60_slope'].apply(lambda x: '日线' + judge_market_scene_by_slope(x))
    daily_df['weekly_scene'] = daily_df['weekly_ma60_slope'].apply(lambda x: '周线' + judge_market_scene_by_slope(x))
    daily_df['combination_scene'] = daily_df.apply(
        lambda x: get_combination_scene(x['weekly_scene'], x['daily_scene']), axis=1
    )
    
    # 初始化回测参数
    capital = initial_capital
    position = 0.0  # 当前仓位价值
    shares = 0.0    # 当前持股数量
    prev_scene = None
    
    # 跟踪M5和M10的状态
    ma5_above_ma10_state = False  # M5是否在M10上方
    golden_cross_confirmed = False  # 金叉是否已确认(保持2天)
    death_cross_confirmed = False  # 死叉是否已确认(保持2天)
    
    # 为每个日期计算策略表现
    daily_results = []
    scene_returns = {scene: {'days': 0, 'total_return': 0.0} for scene in scene_configs.keys()}
    
    # 等待期标记 - 直到周线M60计算完成后才开始交易
    waiting_period = True
    
    # 跟踪最早的有效周线M60日期
    first_valid_weekly_date = None
    if not weekly_df.empty and 'date' in weekly_df.columns:
        first_valid_weekly_date = weekly_df['date'].iloc[0] if len(weekly_df) > 0 else None
        
    # 重置交易记录（确保每次回测都是全新的记录）
    global all_trades
    all_trades = []
    
    for i, row in daily_df.iterrows():
        date = row['date']
        close_price = row['close']
        current_scene = row['combination_scene']
        ma5_above_ma10_days = row.get('ma5_above_ma10_days', 0)
        ma5_below_ma10_days = row.get('ma5_below_ma10_days', 0)
        
        # 检查等待期是否结束（当日期超过最早的有效周线M60日期）
        if waiting_period and first_valid_weekly_date and date >= first_valid_weekly_date:
            waiting_period = False
            # 等待期结束时重置仓位，确保能正确初始化
            position = 0.0
            shares = 0.0
            print(f"周线M60计算完成，等待期结束 ({date})，开始交易")
        
        # 如果是第一天，初始化参数
        if i == 0:
            prev_scene = current_scene
            daily_results.append({
                'date': date,
                'close_price': close_price,
                'scene': current_scene,
                'capital': capital,
                'position': position,
                'total_assets': capital + position,
                'return_rate': 0.0
            })
            continue
        
        # 确保daily_results不为空
        if not daily_results:
            # 如果列表为空，添加初始数据
            daily_results.append({
                'date': date, 
                'close_price': close_price,
                'scene': current_scene,
                'capital': initial_capital,
                'position': 0.0,
                'total_assets': initial_capital,
                'return_rate': 0.0
            })
            continue
        
        # 更新当前仓位价值
        if waiting_period:
            # 等待期内保持空仓
            position = 0.0
            shares = 0.0
            capital = initial_capital
            total_assets = capital + position
        else:
            # 等待期后正常更新仓位价值
            position = shares * close_price
            total_assets = capital + position
        
        # 计算日收益率
        prev_assets = daily_results[-1]['total_assets']
        daily_return = (total_assets - prev_assets) / prev_assets * 100
        
        # 记录当前场景的表现
        if current_scene in scene_returns:
            scene_returns[current_scene]['days'] += 1
            scene_returns[current_scene]['total_return'] += daily_return
        
        # 场景变化时的操作
        scene_changed = current_scene != prev_scene
        
        # 检查M5和M10的状态，确认金叉/死叉
        if ma5_above_ma10_days >= 2:
            golden_cross_confirmed = True
            death_cross_confirmed = False
        elif ma5_below_ma10_days >= 2:
            death_cross_confirmed = True
            golden_cross_confirmed = False
        
        # 等待期结束后才执行交易操作
        if not waiting_period:
            # 记录场景转换
            if scene_changed:
                # 记录场景转换
                if prev_scene is not None:
                    all_trades.append({
                        'date': date,
                        'prev_scene': prev_scene,
                        'current_scene': current_scene,
                        'action': '场景转换',
                        'price': close_price,
                        'shares': shares,
                        'capital': capital,
                        'position': position
                    })
                prev_scene = current_scene
            
            # 移除独立的初始化仓位逻辑，所有建仓操作都严格按照阶梯加仓逻辑处理
            # 确保多头少阴场景也不能自动建仓
            
            # 根据当前场景和M5/M10交叉状态进行阶梯加仓/减仓操作
            if current_scene in scene_configs:
                config = scene_configs[current_scene]
                target_position_ratio = config['target_position']
                current_position_ratio = position / total_assets if total_assets > 0 else 0
            
            # 阶梯加仓操作：M5上穿M10且保持2天
            if golden_cross_confirmed:
                # 只有在非震荡场景且非多头少阴、非空头少阴场景下才执行加仓
                is_non_shock_scene = current_scene not in ['震荡小阳', '震荡平衡', '震荡小阴']
                if is_non_shock_scene and current_scene != '多头少阴' and current_scene != '空头少阴':
                    # 多头场景下使用较高的加仓比例
                    increase_rate = config['golden_cross_increase_rate']
                    
                    # 计算目标仓位
                    target_after_increase = min(1.0, current_position_ratio + increase_rate)
                    buy_amount = total_assets * (target_after_increase - current_position_ratio)
                    
                    if buy_amount > 0 and capital >= buy_amount:
                        capital -= buy_amount
                        shares_added = int(buy_amount / close_price / 100) * 100  # 股份取整为100的倍数
                        shares += shares_added
                        position = shares * close_price
                        all_trades.append({
                            'date': date,
                            'scene': current_scene,
                            'action': '阶梯加仓',
                            'price': close_price,
                            'amount': buy_amount,
                            'shares_added': shares_added,
                            'capital': capital,
                            'position': position,
                            'reason': f'M5在M10上方保持{ma5_above_ma10_days}天'
                        })
                        # 重置金叉确认状态，避免重复加仓
                        golden_cross_confirmed = False
                
            # 阶梯减仓操作：M5下穿M10且保持2天
            elif death_cross_confirmed:
                # 只有在非震荡场景下才执行减仓（包括多头少阴场景）
                is_non_shock_scene = current_scene not in ['震荡小阳', '震荡平衡', '震荡小阴']
                if is_non_shock_scene:
                    # 根据场景使用不同的减仓比例
                    decrease_rate = config['death_cross_decrease_rate']
                    
                    # 计算目标仓位
                    target_after_decrease = max(target_position_ratio, current_position_ratio - decrease_rate)
                    sell_amount = total_assets * (current_position_ratio - target_after_decrease)
                    
                    if sell_amount > 0 and shares > 0:
                        shares_to_sell = min(shares, int(sell_amount / close_price / 100) * 100)  # 股份取整为100的倍数
                        capital += shares_to_sell * close_price
                        shares -= shares_to_sell
                        position = shares * close_price
                        all_trades.append({
                            'date': date,
                            'scene': current_scene,
                            'action': '阶梯减仓',
                            'price': close_price,
                            'amount': shares_to_sell * close_price,
                            'shares_sold': shares_to_sell,
                            'capital': capital,
                            'position': position,
                            'reason': f'M5在M10下方保持{ma5_below_ma10_days}天'
                        })
                        # 重置死叉确认状态，避免重复减仓
                        death_cross_confirmed = False
            
            # 基础的仓位调整：确保仓位在目标范围内
            elif current_position_ratio < config['buy_threshold']:
                buy_amount = total_assets * (target_position_ratio - current_position_ratio)
                if buy_amount > 0 and capital >= buy_amount:
                    capital -= buy_amount
                    shares_added = int(buy_amount / close_price / 100) * 100  # 股份取整为100的倍数
                    shares += shares_added
                    position = shares * close_price
                    all_trades.append({
                        'date': date,
                        'scene': current_scene,
                        'action': '买入',
                        'price': close_price,
                        'amount': buy_amount,
                        'shares_added': shares_added,
                        'capital': capital,
                        'position': position
                    })
            
            elif current_position_ratio > config['sell_threshold']:
                sell_amount = total_assets * (current_position_ratio - target_position_ratio)
                if sell_amount > 0 and shares > 0:
                    shares_to_sell = min(shares, int(sell_amount / close_price / 100) * 100)  # 股份取整为100的倍数
                    capital += shares_to_sell * close_price
                    shares -= shares_to_sell
                    position = shares * close_price
                    all_trades.append({
                        'date': date,
                        'scene': current_scene,
                        'action': '卖出',
                        'price': close_price,
                        'amount': shares_to_sell * close_price,
                        'shares_sold': shares_to_sell,
                        'capital': capital,
                        'position': position
                    })
        
        # 记录每日结果
        daily_results.append({
            'date': date,
            'close_price': close_price,
            'scene': current_scene,
            'capital': capital,
            'position': position,
            'total_assets': total_assets,
            'return_rate': daily_return
        })
    
    # 计算分场景的平均收益率
    for scene in scene_returns:
        if scene_returns[scene]['days'] > 0:
            scene_returns[scene]['avg_return'] = scene_returns[scene]['total_return'] / scene_returns[scene]['days']
        else:
            scene_returns[scene]['avg_return'] = 0.0
    
    # 转换为DataFrame
    results_df = pd.DataFrame(daily_results)
    
    # 计算总收益率
    total_return = (results_df['total_assets'].iloc[-1] - initial_capital) / initial_capital * 100
    
    return results_df, scene_returns, total_return, all_trades

def generate_strategy_report(stock_code, start_date, end_date, results_df, scene_returns, total_return, trades=None):
    """
    生成策略回测报告
    trades: 交易记录列表，如果为None则使用全局的all_trades
    """
    # 使用传入的交易记录或全局交易记录
    global all_trades
    display_trades = trades if trades is not None else all_trades
    """
    生成策略回测报告
    """
    # 报告标题
    add_markdown(f"# 万科A ({stock_code}) 斜率策略回测报告")
    add_markdown(f"\n## 回测时间范围")
    add_markdown(f"{start_date} 至 {end_date}")
    
    # 数据概览
    add_markdown(f"\n## 数据概览")
    add_markdown(f"- 交易日数量: {len(results_df)}")
    add_markdown(f"- 初始资金: {results_df['capital'].iloc[0]:,.2f}")
    add_markdown(f"- 最终资产: {results_df['total_assets'].iloc[-1]:,.2f}")
    add_markdown(f"- 总收益率: {total_return:.2f}%")
    
    # 各场景配置
    add_markdown(f"\n## 9种组合场景策略配置")
    add_markdown(f"| 组合场景名称 | 目标仓位 | 买入阈值 | 卖出阈值 | 调整步长 | 市场特征 |")
    add_markdown(f"|------------|---------|---------|---------|---------|---------|")
    for scene, config in scene_configs.items():
        add_markdown(f"| {scene} | {config['target_position']*100:.0f}% | {config['buy_threshold']*100:.0f}% | {config['sell_threshold']*100:.0f}% | {config['step_size']*100:.0f}% | {config['description']} |")
    
    # 场景加建仓逻辑
    add_markdown(f"\n## 场景加建仓逻辑详解")
    add_markdown(f"策略采用双逻辑结合的仓位管理方法：基础仓位控制 + M5/M10均线交叉阶梯加减仓")
    add_markdown(f"\n### 基础仓位控制")
    add_markdown(f"1. **买入条件**：当当前仓位比例低于目标场景的买入阈值时，进行买入操作")
    add_markdown(f"2. **卖出条件**：当当前仓位比例高于目标场景的卖出阈值时，进行卖出操作")
    add_markdown(f"3. **调整金额计算**：根据总资金、目标仓位和当前仓位的差值计算调整金额")
    
    add_markdown(f"\n### M5/M10均线交叉阶梯加减仓策略")
    add_markdown(f"1. **阶梯加仓触发条件**：M5上穿M10且保持M5>M10状态2天及以上")
    add_markdown(f"2. **阶梯减仓触发条件**：M5下穿M10且保持M5<M10状态2天及以上")
    add_markdown(f"3. **阶梯比例差异化**：根据不同市场场景设置不同的加仓/减仓比例：")
    add_markdown(f"   - 多头场景：较高的加仓比例，较低的减仓比例")
    add_markdown(f"   - 震荡场景：中等的加仓/减仓比例")
    add_markdown(f"   - 空头场景：较低的加仓比例，较高的减仓比例")
    add_markdown(f"4. **单次操作限制**：加仓后仓位不超过100%，减仓后仓位不低于该场景的目标仓位")
    
    add_markdown(f"\n### 各场景阶梯加减仓比例配置")
    add_markdown(f"| 组合场景名称 | 目标仓位 | 金叉加仓比例 | 死叉减仓比例 | 市场特征 |")
    add_markdown(f"|------------|---------|------------|------------|---------|")
    for scene, config in scene_configs.items():
        add_markdown(f"| {scene} | {config['target_position']*100:.0f}% | {config['golden_cross_increase_rate']*100:.0f}% | {config['death_cross_decrease_rate']*100:.0f}% | {config['description']} |")
    
    add_markdown(f"\n### 具体各场景的加建仓操作特征")
    add_markdown(f"- **多头太阳**：保持高仓位（90%），金叉时加仓15%，死叉时减仓10%")
    add_markdown(f"- **多头少阳**：保持中等仓位（70%），金叉时加仓12%，死叉时减仓8%")
    add_markdown(f"- **多头少阴**：保持较低仓位（50%），金叉时加仓10%，死叉时减仓6%")
    add_markdown(f"- **震荡小阳**：保持适度仓位（60%），金叉时加仓8%，死叉时减仓8%")
    add_markdown(f"- **震荡平衡**：保持轻仓（40%），金叉时加仓6%，死叉时减仓6%")
    add_markdown(f"- **震荡小阴**：保持轻仓（30%），金叉时加仓5%，死叉时减仓9%")
    add_markdown(f"- **空头小阳**：保持极低仓位（20%），金叉时加仓4%，死叉时减仓12%")
    add_markdown(f"- **空头少阴**：保持极小仓位（10%），金叉时加仓3%，死叉时减仓15%")
    add_markdown(f"- **空头太阴**：保持空仓（0%），不进行任何买入操作，死叉时减仓20%")
    
    # 分场景收益率
    add_markdown(f"\n## 分场景收益率分析")
    add_markdown(f"| 组合场景名称 | 天数 | 总收益率(%) | 日均收益率(%) |")
    add_markdown(f"|------------|-----|------------|------------|")
    for scene, stats in scene_returns.items():
        add_markdown(f"| {scene} | {stats['days']} | {stats['total_return']:.2f} | {stats['avg_return']:.4f} |")
    
    # 资产曲线图
    add_markdown(f"\n## 资产曲线走势")
    # 这里可以添加资产曲线的图表代码
    
    # 交易记录统计
    add_markdown(f"\n## 交易记录统计")
    add_markdown(f"- 总交易次数: {len(display_trades)}")
    buy_trades = [trade for trade in display_trades if trade.get('action') == '买入']
    sell_trades = [trade for trade in display_trades if trade.get('action') == '卖出']
    scene_changes = [trade for trade in display_trades if trade.get('action') == '场景转换']
    add_markdown(f"- 买入次数: {len(buy_trades)}")
    add_markdown(f"- 卖出次数: {len(sell_trades)}")
    add_markdown(f"- 场景转换次数: {len(scene_changes)}")
    
    # 交易明细表格
    add_markdown(f"\n## 交易明细")
    if display_trades:
        add_markdown(f"| 交易日期 | 场景 | 操作类型 | 价格 | 金额 | 股份变动 | 剩余资金 | 持仓价值 | 持仓股份 | 交易原因 |")
        add_markdown(f"|--------|-----|--------|------|------|--------|--------|--------|--------|--------|")
        for trade in display_trades:
            action = trade.get('action')
            date = trade.get('date', '')
            scene = trade.get('scene', '')
            price = trade.get('price', 0)
            amount = trade.get('amount', 0)
            # 处理买入和阶梯加仓的股份变动
            shares_change = int(trade.get('shares_added', 0)) if action in ['买入', '阶梯加仓'] else int(trade.get('shares_sold', 0))  # 股份显示为整数
            capital = trade.get('capital', 0)
            position = trade.get('position', 0)
            reason = trade.get('reason', '')
            
            # 场景转换的特殊处理
            if action == '场景转换':
                prev_scene = trade.get('prev_scene', '')
                current_scene = trade.get('current_scene', '')
                scene = f"{prev_scene}→{current_scene}"
                shares_change = '-'  # 场景转换不涉及股份变动
                # 为场景转换设置特殊的价格和金额显示
                price_str = '-'
                amount_str = '-'
                capital_str = f"{capital:,.2f}"
                position_str = f"{position:,.2f}"
                # 场景转换的持仓股份
                shares_held = int(trade.get('shares', 0)) if price > 0 else 0
                shares_held_str = f"{shares_held:,}"
                reason = '市场场景转换'
            else:
                # 正常交易的格式化
                price_str = f"{price:.2f}"
                amount_str = f"{amount:.2f}"
                capital_str = f"{capital:,.2f}"
                position_str = f"{position:,.2f}"
                # 计算持仓股份（持仓价值除以价格）
                shares_held = int(position / price) if price > 0 else 0
                shares_held_str = f"{shares_held:,}"
                if action == '阶梯加仓' and not reason:
                    reason = 'M5上穿M10且保持2天以上'
                elif action == '阶梯减仓' and not reason:
                    reason = 'M5下穿M10且保持2天以上'
                elif action == '买入' and not reason:
                    reason = '仓位低于买入阈值'
                elif action == '卖出' and not reason:
                    reason = '仓位高于卖出阈值'
            
            add_markdown(f"| {date} | {scene} | {action} | {price_str} | {amount_str} | {shares_change} | {capital_str} | {position_str} | {shares_held_str} | {reason} |")
    else:
        add_markdown(f"暂无交易记录")
    
    # 策略效果分析
    add_markdown(f"\n## 策略效果分析")
    add_markdown(f"- 策略在多头场景下表现: {'优秀' if any(stats['avg_return'] > 0 for scene, stats in scene_returns.items() if '多头' in scene) else '一般'}")
    add_markdown(f"- 策略在空头场景下表现: {'良好' if any(stats['avg_return'] > -0.1 for scene, stats in scene_returns.items() if '空头' in scene) else '待改进'}")
    add_markdown(f"- 策略在震荡场景下表现: {'稳定' if any(abs(stats['avg_return']) < 0.2 for scene, stats in scene_returns.items() if '震荡' in scene) else '波动较大'}")
    
    # 策略优化建议
    add_markdown(f"\n## 策略优化建议")
    add_markdown(f"1. 可以根据不同股票的特性调整各场景的目标仓位和阶梯加减仓比例")
    add_markdown(f"2. 考虑调整M5和M10交叉后的确认天数参数（当前为2天）")
    add_markdown(f"3. 针对不同市场环境，可以动态调整阶梯加减仓的比例")
    add_markdown(f"4. 设置止损止盈机制，控制单笔交易的最大亏损和最大盈利")
    add_markdown(f"5. 考虑引入更多技术指标作为辅助判断，如MACD、RSI等")
    add_markdown(f"6. 测试不同的均线组合（如M10/M20、M20/M60等）以寻找最优参数组合")
    add_markdown(f"7. 在震荡市中，可以考虑增加T+0策略以提高资金利用率")

def main():
    # 设置回测时间范围（最近10年）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    stock_code = "sz.000002"  # 万科A
    
    # 获取数据
    daily_df, weekly_df = get_kline_data(start_date, end_date, stock_code)
    
    if daily_df is None or weekly_df is None:
        print("数据获取失败，无法继续回测")
        return
    
    # 计算日线60日均线斜率
    daily_df = calculate_slope(daily_df, window=10, period=60)
    
    # 计算周线60周均线斜率
    weekly_df = calculate_slope(weekly_df, window=5, period=60)
    
    # 过滤掉NaN值
    daily_df = daily_df.dropna(subset=['ma60_slope'])
    weekly_df = weekly_df.dropna(subset=['ma60_slope'])
    
    # 进行策略回测
    results_df, scene_returns, total_return, all_trades = backtest_strategy(daily_df, weekly_df)
    
    # 生成报告
    generate_strategy_report(stock_code, start_date, end_date, results_df, scene_returns, total_return, all_trades)
    
    # 保存报告
    save_to_markdown(stock_code)

if __name__ == "__main__":
    main()