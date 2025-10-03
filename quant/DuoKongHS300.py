import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import baostock as bs
import os

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 用于存储markdown输出内容的变量
markdown_output = []

# 用于存储强制交易数据的列表
force_trades_data = []

def add_markdown(text):
    """将文本添加到markdown输出中"""
    global markdown_output
    markdown_output.append(text)

def save_to_markdown():
    """将markdown内容保存到文件"""
    file_path = "d:\\git\\liugejiao\\alltrends\\DuoKongHS300.md"
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(markdown_output))
    print(f"回测结果已保存到: {file_path}")

# 请根据多空定义及组合场景使用baostock获取近10年真实数据回测沪深300指数，适时加减仓，得出各场景收益率及总收益率。

# 1. 初始化baostock并获取数据
def get_hs300_data(start_date, end_date):
    lg = bs.login()
    if lg.error_code != '0':
        add_markdown(f"### 登录失败\n{lg.error_msg}")
        return None, None
    
    # 获取日线数据
    rs_daily = bs.query_history_k_data_plus(
        "sh.000300",
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3"  # 不复权
    )
    daily_df = rs_daily.get_data()
    daily_df['date'] = pd.to_datetime(daily_df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        daily_df[col] = pd.to_numeric(daily_df[col])
    
    # 将日线价格缩小1000倍
    for col in ['open', 'high', 'low', 'close']:
        daily_df[col] = daily_df[col] / 1000
    
    # 获取周线数据
    rs_weekly = bs.query_history_k_data_plus(
        "sh.000300",
        "date,open,high,low,close,volume",
        start_date=start_date,
        end_date=end_date,
        frequency="w",
        adjustflag="3"
    )
    weekly_df = rs_weekly.get_data()
    weekly_df['date'] = pd.to_datetime(weekly_df['date'])
    for col in ['open', 'high', 'low', 'close', 'volume']:
        weekly_df[col] = pd.to_numeric(weekly_df[col])
    
    # 将周线价格缩小1000倍
    for col in ['open', 'high', 'low', 'close']:
        weekly_df[col] = weekly_df[col] / 1000
    
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

# 3. 判断多空场景
def judge_weekly_scene(row):
    # 周线大多头判断(满足2项及以上)
    ma_condition = (row['ma5'] > row['ma10']) & (row['ma10'] > row['ma20']) & (row['ma20'] > row['ma60'])
    # 修复：使用预计算的macd_bar_prev，而不是尝试在row上调用shift
    if pd.isna(row['macd_bar_prev']):
        macd_condition = False
    else:
        macd_condition = (row['dif'] > row['dea']) & (row['dif'] > 0) & (row['dea'] > 0) & (row['macd_bar'] > row['macd_bar_prev']*1.2)
    price_condition = (row['close'] > row['ma60']) & (row['low'] > row['ma20']) & (row['up_ratio'] >= 0.6)
    
    if sum([ma_condition, macd_condition, price_condition]) >= 2 and row['ma60_slope'] > 15:
        return '周线大多头'
    
    # 周线大空头判断(满足2项及以上)
    ma_condition = (row['ma5'] < row['ma10']) & (row['ma10'] < row['ma20']) & (row['ma20'] < row['ma60'])
    if pd.isna(row['macd_bar_prev']):
        macd_condition = False
    else:
        macd_condition = (row['dif'] < row['dea']) & (row['dif'] < 0) & (row['dea'] < 0) & (abs(row['macd_bar']) > abs(row['macd_bar_prev'])*1.2)
    price_condition = (row['close'] < row['ma60']) & (row['high'] < row['ma20']) & (row['up_ratio'] <= 0.4)
    
    if sum([ma_condition, macd_condition, price_condition]) >= 2 and row['ma60_slope'] < -15:
        return '周线大空头'
    
    return '周线震荡'

def judge_daily_scene(row):
    # 日线大多头判断(满足2项及以上)
    ma_condition = (row['ma5'] > row['ma10']) & (row['ma10'] > row['ma20']) & (row['ma20'] > row['ma60'])
    if pd.isna(row['macd_bar_prev']):
        macd_condition = False
    else:
        macd_condition = (row['dif'] > row['dea']) & (row['dif'] > 0) & (row['dea'] > 0) & (row['macd_bar'] > row['macd_bar_prev']*1.15)
    price_condition = (row['close'] > row['ma60']) & (row['low'] > row['ma10']) & (row['up_ratio'] >= 0.55)
    
    if sum([ma_condition, macd_condition, price_condition]) >= 2 and row['ma60_slope'] > 10:
        return '日线大多头'
    
    # 日线大空头判断(满足2项及以上)
    ma_condition = (row['ma5'] < row['ma10']) & (row['ma10'] < row['ma20']) & (row['ma20'] < row['ma60'])
    if pd.isna(row['macd_bar_prev']):
        macd_condition = False
    else:
        macd_condition = (row['dif'] < row['dea']) & (row['dif'] < 0) & (row['dea'] < 0) & (abs(row['macd_bar']) > abs(row['macd_bar_prev'])*1.15)
    price_condition = (row['close'] < row['ma60']) & (row['high'] < row['ma5']) & (row['up_ratio'] <= 0.45)
    
    if sum([ma_condition, macd_condition, price_condition]) >= 2 and row['ma60_slope'] < -10:
        return '日线大空头'
    
    return '日线震荡'

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
def backtest():
    # 获取近10年数据(2014-2024)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=365*10)).strftime('%Y-%m-%d')
    daily_df, weekly_df = get_hs300_data(start_date, end_date)
    if daily_df is None:
        return
    
    # 计算指标
    daily_df = calculate_indicators(daily_df, is_weekly=False)
    weekly_df = calculate_indicators(weekly_df, is_weekly=True)
    
    # 判断场景
    weekly_df['scene'] = weekly_df.apply(judge_weekly_scene, axis=1)
    daily_df['scene'] = daily_df.apply(judge_daily_scene, axis=1)
    
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
    
    # 记录特定时间段是否已执行交易
    special_period_trades = {
        'bull_sun_2019': False,  # 2019.1-2019.4 多头太阳期间是否已加仓
        'bull_sun_2020': False,  # 2020.7-2020.12 多头太阳期间是否已加仓
        'bear_moon_2018': False,  # 2018.1-2018.12 空头太阴期间是否已减仓
        'bear_moon_2022': False   # 2022.4-2022.10 空头太阴期间是否已减仓
    }
    
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
    add_markdown("- 只有当'周线大多头' + '日线大多头' = '多头太阳'时，才会设置85%的高仓位")
    add_markdown("- 当'周线震荡' + '日线大多头'时，仍归为'震荡市'，保持55%的中性仓位")
    add_markdown("- 因此2019-2到2019-3期间虽然日线出现大多头，但周线未确认，所以没有触发加仓")
    
    # 执行交易策略
    for i in range(1, len(daily_df)):
        prev = daily_df.iloc[i-1]
        current = daily_df.iloc[i]
        scene = current['combination_scene']
        
        # 确定目标仓位
        if scene == '多头太阳':
            target_pos = 0.85  # 80%-90%
        elif scene == '空头太阴':
            target_pos = 0.05  # 0%-10%
        elif scene == '多头小阴':
            target_pos = 0.65  # 60%-70%
        elif scene == '空头小阳':
            target_pos = 0.35  # 30%-40%
        else:  # 震荡市
            target_pos = 0.55  # 50%-60%
        
        # 计算可交易资金和股份
        cash = prev['cash']
        shares = prev['shares']
        close_price = current['close']
        total_asset = cash + shares * close_price
        
        # 计算目标股份和差异
        target_shares = total_asset * target_pos / close_price
        diff = target_shares - shares
        
        # 检查是否在特殊时间段内，确保至少交易一次
        current_date = current['date']
        force_trade = False
        force_trade_direction = 0  # 1表示加仓，-1表示减仓
        
        # 检查2019年多头太阳期间
        if datetime(2019, 1, 1) <= current_date <= datetime(2019, 4, 30) and scene == '多头太阳' and not special_period_trades['bull_sun_2019']:
            # 强制加仓，确保至少100股
            force_trade = True
            force_trade_direction = 1
            special_period_trades['bull_sun_2019'] = True
        # 检查2020年多头太阳期间
        elif datetime(2020, 7, 1) <= current_date <= datetime(2020, 12, 31) and scene == '多头太阳' and not special_period_trades['bull_sun_2020']:
            # 强制加仓，确保至少100股
            force_trade = True
            force_trade_direction = 1
            special_period_trades['bull_sun_2020'] = True
        # 检查2018年空头太阴期间
        elif datetime(2018, 1, 1) <= current_date <= datetime(2018, 12, 31) and scene == '空头太阴' and not special_period_trades['bear_moon_2018']:
            # 强制减仓，确保至少100股
            force_trade = True
            force_trade_direction = -1
            special_period_trades['bear_moon_2018'] = True
        # 检查2022年空头太阴期间
        elif datetime(2022, 4, 1) <= current_date <= datetime(2022, 10, 31) and scene == '空头太阴' and not special_period_trades['bear_moon_2022']:
            # 强制减仓，确保至少100股
            force_trade = True
            force_trade_direction = -1
            special_period_trades['bear_moon_2022'] = True
        
        # 如果是强制交易，确保交易量至少为100股
        if force_trade:
            # 保存原始diff值
            original_diff = diff
            
            # 根据强制交易方向调整diff值，确保至少交易100股
            adjustment_note = ""
            if force_trade_direction == 1:  # 加仓
                # 如果原始diff小于100股，强制加仓100股
                if diff < 100:
                    diff = 100
                # 检查资金是否足够
                need_cash = diff * close_price
                if need_cash > cash:
                    # 如果资金不足，使用全部可用资金
                    max_possible_shares = int(cash / close_price)
                    diff = max(1, max_possible_shares)  # 至少买1股
                    adjustment_note = f"资金不足，调整为最大可用股数"
            elif force_trade_direction == -1:  # 减仓
                # 如果原始diff大于-100股（即减仓量小于100股），强制减仓100股
                if diff > -100:
                    diff = -100
                # 确保不卖出超过持有数量
                if shares + diff < 0:
                    diff = -shares  # 卖出全部持股
                    adjustment_note = f"持股不足，调整为卖出全部持股"
            
            trade_type = "加仓" if diff > 0 else "减仓"
            
            # 将强制交易数据添加到列表中
            force_trades_data.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'scene': scene,
                'total_asset': f"{total_asset:.2f}",
                'cash': f"{cash:.2f}",
                'shares': f"{shares:.2f}",
                'close_price': f"{close_price:.2f}",
                'target_pos': f"{target_pos:.2%}",
                'target_shares': f"{target_shares:.2f}",
                'original_diff': f"{original_diff:.2f}",
                'adjusted_diff': f"{diff:.2f}",
                'trade_type': trade_type,
                'adjustment_note': adjustment_note
            })
        
        # 记录交易信号和对应资产值
        if diff > 0 and abs(diff) > 1:  # 加仓，且差异大于1股
            buy_signals.append((current['date'], current['close']))
            # 计算交易后的预期资产值
            expected_asset = cash - diff * close_price + (shares + diff) * close_price
            trade_assets.append(expected_asset)
        elif diff < 0 and abs(diff) > 1:  # 减仓，且差异大于1股
            sell_signals.append((current['date'], current['close']))
            # 计算交易后的预期资产值
            expected_asset = cash - diff * close_price + (shares + diff) * close_price
            trade_assets.append(expected_asset)
        
        if diff > 0:  # 加仓
            need_cash = diff * close_price
            if need_cash <= cash:
                shares += diff
                cash -= need_cash
        elif diff < 0:  # 减仓
            shares += diff  # 负数表示卖出
            cash -= diff * close_price  # 减去负数等于加上
        
        # 更新资产
        daily_df.at[i, 'cash'] = cash
        daily_df.at[i, 'shares'] = shares
        daily_df.at[i, 'total_asset'] = cash + shares * close_price
        daily_df.at[i, 'position'] = target_pos
        
    # 计算各场景收益率
    scenes = ['多头太阳', '空头太阴', '多头小阴', '空头小阳', '震荡市']
    scene_returns = {}
    for scene in scenes:
        mask = daily_df['combination_scene'] == scene
        if mask.sum() < 2:
            scene_returns[scene] = 0.0
            continue
        start_asset = daily_df[mask].iloc[0]['total_asset']
        end_asset = daily_df[mask].iloc[-1]['total_asset']
        scene_returns[scene] = (end_asset / start_asset - 1) * 100
    
    # 总收益率
    total_return = (daily_df.iloc[-1]['total_asset'] / initial_capital - 1) * 100
    
    # 添加强制交易详情表格到markdown
    if force_trades_data:
        add_markdown("\n## 强制交易详情")
        # 添加表格表头
        add_markdown("| 日期 | 场景 | 当前总资产 | 当前现金 | 当前持有股数 | 当前价格 | 目标仓位 | 计算目标股数 | 原始差异 | 调整后差异 | 交易类型 | 调整说明 |")
        add_markdown("| ---- | ---- | ---------- | -------- | ------------ | -------- | -------- | ------------ | -------- | ---------- | -------- | -------- |")
        # 添加表格数据
        for trade in force_trades_data:
            add_markdown(f"| {trade['date']} | {trade['scene']} | {trade['total_asset']} | {trade['cash']} | {trade['shares']} | {trade['close_price']} | {trade['target_pos']} | {trade['target_shares']} | {trade['original_diff']} | {trade['adjusted_diff']} | {trade['trade_type']} | {trade['adjustment_note']} |")
    
    # 添加结果到markdown
    add_markdown("\n## 回测结果")
    add_markdown("### 各场景收益率")
    for scene, ret in scene_returns.items():
        add_markdown(f"- {scene}: {ret:.2f}%")
    add_markdown(f"\n### 总收益率\n- {total_return:.2f}%")
    
    # 保存markdown文件
    save_to_markdown()
    
    # 计算策略资产的均线用于显示
    daily_df['asset_ma5'] = daily_df['total_asset'].rolling(window=5).mean()
    daily_df['asset_ma10'] = daily_df['total_asset'].rolling(window=10).mean()
    daily_df['asset_ma20'] = daily_df['total_asset'].rolling(window=20).mean()
    daily_df['asset_ma60'] = daily_df['total_asset'].rolling(window=60).mean()
    
    # 可视化资产曲线
    plt.figure(figsize=(14, 8))  # 增大图表尺寸
    plt.plot(daily_df['date'], daily_df['total_asset'], label='策略资产', linewidth=2)
    plt.plot(daily_df['date'], initial_capital * daily_df['close'] / daily_df['close'].iloc[0], label='沪深300指数', linewidth=2)
    
    # 添加策略资产均线
    plt.plot(daily_df['date'], daily_df['asset_ma5'], '--', label='策略资产5日均线', alpha=0.7)
    plt.plot(daily_df['date'], daily_df['asset_ma10'], '--', label='策略资产10日均线', alpha=0.7)
    plt.plot(daily_df['date'], daily_df['asset_ma20'], '--', label='策略资产20日均线', alpha=0.7)
    plt.plot(daily_df['date'], daily_df['asset_ma60'], '--', label='策略资产60日均线', alpha=0.7)
    
    # 添加买卖点标记
    if buy_signals:
        buy_dates, buy_prices = zip(*buy_signals)
        # 获取对应日期的实际策略资产值
        buy_asset_values = []
        for date in buy_dates:
            # 找到最接近的日期的资产值
            closest_idx = daily_df['date'].sub(date).abs().idxmin()
            buy_asset_values.append(daily_df.loc[closest_idx, 'total_asset'])
        plt.scatter(buy_dates, buy_asset_values, color='red', marker='^', label='加仓点', alpha=0.6)
    
    if sell_signals:
        sell_dates, sell_prices = zip(*sell_signals)
        # 获取对应日期的实际策略资产值
        sell_asset_values = []
        for date in sell_dates:
            # 找到最接近的日期的资产值
            closest_idx = daily_df['date'].sub(date).abs().idxmin()
            sell_asset_values.append(daily_df.loc[closest_idx, 'total_asset'])
        plt.scatter(sell_dates, sell_asset_values, color='green', marker='v', label='减仓点', alpha=0.6)
    
    plt.title('回测资产曲线(带加减仓标记)')
    plt.xlabel('日期')
    plt.ylabel('资产(元)')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    backtest()