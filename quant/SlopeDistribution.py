# @brief: 计算股票的斜率分布，参见文件[AllTrendsStrategy.md] 中的详细说明
# 采用微软qlib库的股票近10年数据，数据目录为d:/data/qlib_bin，计算斜率，支持指定股票代码，
# 根据核心思想：周线定方向，日线找买点
# - 周线定方向：确保我们只在“大势”向上的情况下操作，提高胜率，避免逆势交易。
# - 日线找买点：在确定的多头方向上，利用日线级别的回调寻找风险收益比更佳的入场点。
# 定义多头：指定30周线或者60周线，如果30周线或60周线的斜率为正，则认为是多头
# 定义空头：指定30周线或者60周线，如果30周线或60周线的斜率为负，则认为是空头
# 根据市场24种组合场景，设计斜率策略，每个组合场景下，根据斜率分布设置目标仓位、加减仓逻辑
# 报告生成在SlopeDistribution_{stock_code}.md文件中

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import os
# 使用qlib库替代baostock
import qlib
from qlib.data import D
# 导入我们的qlib数据获取器
from qlib_data_fetcher import QlibDataFetcher

# 设置matplotlib支持中文显示
plt.rcParams["font.family"] = ["Microsoft YaHei", "SimHei", "sans-serif"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 用于存储markdown输出内容的变量
markdown_output = []

def add_markdown(text):
    """将文本添加到markdown输出中"""
    global markdown_output
    markdown_output.append(text)

def save_to_markdown(stock_code="sz.000002"):
    """将markdown内容保存到文件"""
    file_path = rf"d:\git\liugejiao\alltrends\SlopeDistribution_{stock_code}.md"
    with open(file_path, 'w', encoding='utf-8-sig') as f:
        f.write('\n'.join(markdown_output))
    print(f"斜率分布分析报告已保存到: {file_path}")

def get_kline_data(start_date, end_date, stock_code="sz.000002"):
    """
    从qlib获取股票的日线和周线数据
    """
    print(f"使用qlib获取{stock_code}的数据...")
    
    # 创建qlib数据获取器
    fetcher = QlibDataFetcher()
    
    # 初始化qlib
    if not fetcher.initialize():
        print("初始化qlib失败")
        return None, None
    
    # 获取日线数据
    print("获取日线数据...")
    daily_df = fetcher.get_daily_data(stock_code, start_date, end_date)
    
    if daily_df is None or daily_df.empty:
        print(f"获取{stock_code}的日线数据失败")
        return None, None
    
    # 获取周线数据
    print("获取周线数据...")
    weekly_df = fetcher.get_weekly_data(stock_code, start_date, end_date)
    
    if weekly_df is None or weekly_df.empty:
        print(f"获取{stock_code}的周线数据失败")
        return None, None
    
    print("数据获取成功!")
    fetcher.close()
    return daily_df, weekly_df

def calculate_slope(df, window=10, period=60):
    """
    计算指定周期均线的斜率分布
    window: 计算斜率的窗口大小
    period: 均线周期
    """
    # 计算均线
    df[f'ma{period}'] = df['close'].rolling(window=period).mean()
    
    # 计算均线斜率(角度)
    df[f'ma{period}_slope'] = np.degrees(np.arctan(
        (df[f'ma{period}'] - df[f'ma{period}'].shift(window)) / window  # window周期斜率
    ))
    
    return df

def analyze_slope_distribution(df, slope_col, title):
    """
    分析斜率分布并生成统计信息和图表
    """
    # 过滤掉NaN值
    valid_slopes = df[slope_col].dropna()
    
    # 计算统计信息
    stats = {
        'count': len(valid_slopes),
        'mean': valid_slopes.mean(),
        'median': valid_slopes.median(),
        'std': valid_slopes.std(),
        'min': valid_slopes.min(),
        'max': valid_slopes.max(),
        'positive_count': (valid_slopes > 0).sum(),
        'negative_count': (valid_slopes < 0).sum(),
        'zero_count': (valid_slopes == 0).sum()
    }
    
    # 计算百分比
    stats['positive_percent'] = stats['positive_count'] / stats['count'] * 100
    stats['negative_percent'] = stats['negative_count'] / stats['count'] * 100
    stats['zero_percent'] = stats['zero_count'] / stats['count'] * 100
    
    # 定义斜率区间
    slope_intervals = [
        ('极陡下跌', (-float('inf'), -5)),
        ('陡峭下跌', (-5, -3)),
        ('中度下跌', (-3, -1)),
        ('轻微下跌', (-1, 0)),
        ('零斜率', (0, 0)),
        ('轻微上涨', (0, 1)),
        ('中度上涨', (1, 3)),
        ('陡峭上涨', (3, 5)),
        ('极陡上涨', (5, float('inf')))
    ]
    
    # 计算各区间的统计数据
    interval_stats = []
    for name, (lower, upper) in slope_intervals:
        if lower == upper:  # 处理零斜率的特殊情况
            count = (valid_slopes == lower).sum()
        else:
            count = ((valid_slopes >= lower) & (valid_slopes < upper)).sum()
        percent = count / stats['count'] * 100 if stats['count'] > 0 else 0
        interval_stats.append({
            'name': name,
            'lower': lower,
            'upper': upper,
            'count': count,
            'percent': percent
        })
    
    stats['interval_stats'] = interval_stats
    
    # 生成直方图
    plt.figure(figsize=(12, 6))
    plt.hist(valid_slopes, bins=50, alpha=0.7, color='blue')
    plt.axvline(stats['mean'], color='red', linestyle='dashed', linewidth=1, label=f'均值: {stats["mean"]:.2f}°')
    plt.axvline(stats['median'], color='green', linestyle='dashed', linewidth=1, label=f'中位数: {stats["median"]:.2f}°')
    plt.title(f'{title}斜率分布直方图')
    plt.xlabel('斜率角度(度)')
    plt.ylabel('频率')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 保存图表
    chart_path = f"slope_distribution_{title.replace(' ', '_')}.png"
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return stats, chart_path

def generate_slope_interval_table(interval_stats):
    """
    生成斜率区间统计的Markdown表格
    """
    table = []
    table.append('| 斜率区间 | 角度范围(度) | 数据点数量 | 占比(%) |')
    table.append('|---------|------------|----------|-------|')
    
    for interval in interval_stats:
        if interval['lower'] == -float('inf'):
            range_str = f'< -{interval["upper"]}'
        elif interval['upper'] == float('inf'):
            range_str = f'> {interval["lower"]}'
        elif interval['lower'] == interval['upper']:
            range_str = f'= {interval["lower"]}'
        else:
            range_str = f'{interval["lower"]} ~ {interval["upper"]}'
        
        table.append(f"| {interval['name']} | {range_str} | {interval['count']} | {interval['percent']:.2f}% |")
    
    return '\n'.join(table)

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


def calculate_combination_scene_stats(daily_df, weekly_df, daily_slope_col='ma60_slope', weekly_slope_col='ma60_slope'):
    """
    计算各组合场景的时间占比
    """
    # 过滤掉NaN值
    daily_df = daily_df.dropna(subset=[daily_slope_col])
    weekly_df = weekly_df.dropna(subset=[weekly_slope_col])
    
    # 为每条日线数据判断场景
    daily_df['scene'] = daily_df[daily_slope_col].apply(lambda x: judge_market_scene_by_slope(x))
    daily_df['scene'] = '日线' + daily_df['scene']
    
    # 合并周线场景到日线数据
    daily_df['weekly_scene'] = daily_df['date'].apply(
        lambda x: weekly_df[weekly_df['date'] <= x][weekly_slope_col].iloc[-1] 
        if not weekly_df[weekly_df['date'] <= x].empty else 0
    )
    
    # 为每条日线数据的周线斜率判断场景
    daily_df['weekly_scene'] = daily_df['weekly_scene'].apply(lambda x: judge_market_scene_by_slope(x))
    daily_df['weekly_scene'] = '周线' + daily_df['weekly_scene']
    
    # 确定组合场景
    daily_df['combination_scene'] = daily_df.apply(
        lambda x: get_combination_scene(x['weekly_scene'], x['scene']), axis=1
    )
    
    # 统计各场景的时间占比
    total_days = len(daily_df)
    scene_counts = daily_df['combination_scene'].value_counts()
    scene_percentages = (scene_counts / total_days * 100).round(2)
    
    # 创建结果DataFrame
    scene_stats = pd.DataFrame({
        '组合场景': scene_counts.index,
        '天数': scene_counts.values,
        '占比(%)': scene_percentages.values
    })
    
    # 按照预设的9种场景顺序排序
    expected_scenes = ['多头太阳', '多头少阳', '多头少阴', '震荡小阳', '震荡平衡', '震荡小阴', '空头小阳', '空头少阴', '空头太阴']
    scene_stats = scene_stats.set_index('组合场景').reindex(expected_scenes).reset_index()
    scene_stats['天数'] = scene_stats['天数'].fillna(0).astype(int)
    scene_stats['占比(%)'] = scene_stats['占比(%)'].fillna(0)
    
    return scene_stats, daily_df


def main():
    # 设置回测时间范围（最近10年）
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=10*365)).strftime('%Y-%m-%d')
    stock_code = "sz.000002"  # 万科A
    
    # 添加报告标题
    add_markdown(f"# 万科A ({stock_code}) 斜率分布分析报告")
    add_markdown(f"\n## 回测时间范围")
    add_markdown(f"{start_date} 至 {end_date}")
    
    # 获取数据
    daily_df, weekly_df = get_kline_data(start_date, end_date, stock_code)
    
    if daily_df is None or weekly_df is None:
        print("数据获取失败，无法继续分析")
        return
    
    # 添加数据概览
    add_markdown(f"\n## 数据概览")
    add_markdown(f"- 日线数据记录数: {len(daily_df)}")
    add_markdown(f"- 周线数据记录数: {len(weekly_df)}")
    
    # 计算日线60日均线斜率
    daily_df = calculate_slope(daily_df, window=10, period=60)
    
    # 计算周线60周均线斜率
    weekly_df = calculate_slope(weekly_df, window=5, period=60)
    
    # 分析日线斜率分布
    daily_stats, daily_chart = analyze_slope_distribution(daily_df, 'ma60_slope', '日线60日均线')
    
    # 分析周线斜率分布
    weekly_stats, weekly_chart = analyze_slope_distribution(weekly_df, 'ma60_slope', '周线60周均线')
    
    # 添加日线斜率分析结果
    add_markdown(f"\n## 日线60日均线斜率分析")
    add_markdown(f"- 有效数据点数量: {daily_stats['count']}")
    add_markdown(f"- 平均斜率: {daily_stats['mean']:.2f}°")
    add_markdown(f"- 中位数斜率: {daily_stats['median']:.2f}°")
    add_markdown(f"- 标准差: {daily_stats['std']:.2f}")
    add_markdown(f"- 最小斜率: {daily_stats['min']:.2f}°")
    add_markdown(f"- 最大斜率: {daily_stats['max']:.2f}°")
    
    # 添加斜率区间统计表格
    add_markdown(f"\n### 日线斜率区间分布统计")
    daily_interval_table = generate_slope_interval_table(daily_stats['interval_stats'])
    add_markdown(daily_interval_table)
    
    add_markdown(f"\n![日线斜率分布图]({daily_chart})")
    
    # 添加周线斜率分析结果
    add_markdown(f"\n## 周线60周均线斜率分析")
    add_markdown(f"- 有效数据点数量: {weekly_stats['count']}")
    add_markdown(f"- 平均斜率: {weekly_stats['mean']:.2f}°")
    add_markdown(f"- 中位数斜率: {weekly_stats['median']:.2f}°")
    add_markdown(f"- 标准差: {weekly_stats['std']:.2f}")
    add_markdown(f"- 最小斜率: {weekly_stats['min']:.2f}°")
    add_markdown(f"- 最大斜率: {weekly_stats['max']:.2f}°")
    
    # 添加斜率区间统计表格
    add_markdown(f"\n### 周线斜率区间分布统计")
    weekly_interval_table = generate_slope_interval_table(weekly_stats['interval_stats'])
    add_markdown(weekly_interval_table)
    
    add_markdown(f"\n![周线斜率分布图]({weekly_chart})")
    
    # 添加斜率分布对比分析
    add_markdown(f"\n## 斜率分布对比分析")
    add_markdown(f"- 日线平均斜率比周线{'高' if daily_stats['mean'] > weekly_stats['mean'] else '低'}: {abs(daily_stats['mean'] - weekly_stats['mean']):.2f}°")
    add_markdown(f"- 日线斜率波动比周线{'大' if daily_stats['std'] > weekly_stats['std'] else '小'}: {abs(daily_stats['std'] - weekly_stats['std']):.2f}")
    
    # 添加周线与日线组合场景分析
    add_markdown(f"\n## 周线与日线组合场景分析")
    add_markdown(f"根据斜率分布，我们可以将市场状态分为9种组合场景：")
    
    # 准备组合场景数据
    # 1. 计算日线和周线的平均斜率
    daily_avg_slope = daily_stats['mean']
    weekly_avg_slope = weekly_stats['mean']
    
    # 2. 判断整体市场场景
    daily_overall_scene = judge_market_scene_by_slope(daily_avg_slope)
    weekly_overall_scene = judge_market_scene_by_slope(weekly_avg_slope)
    
    # 3. 获取当前组合场景
    current_combination = get_combination_scene(f'周线{weekly_overall_scene}', f'日线{daily_overall_scene}')
    
    # 4. 统计各斜率区间对应的场景特征
    daily_slope_scenes = {
        '空头': {'lower': -float('inf'), 'upper': -1, 'count': daily_stats['negative_count'] - (daily_stats['interval_stats'][3]['count']), 'percent': (daily_stats['negative_count'] - daily_stats['interval_stats'][3]['count']) / daily_stats['count'] * 100},
        '震荡': {'lower': -1, 'upper': 1, 'count': daily_stats['zero_count'] + daily_stats['interval_stats'][3]['count'] + daily_stats['interval_stats'][5]['count'], 'percent': (daily_stats['zero_count'] + daily_stats['interval_stats'][3]['count'] + daily_stats['interval_stats'][5]['count']) / daily_stats['count'] * 100},
        '多头': {'lower': 1, 'upper': float('inf'), 'count': daily_stats['positive_count'] - daily_stats['interval_stats'][5]['count'], 'percent': (daily_stats['positive_count'] - daily_stats['interval_stats'][5]['count']) / daily_stats['count'] * 100}
    }
    
    weekly_slope_scenes = {
        '空头': {'lower': -float('inf'), 'upper': -1, 'count': weekly_stats['negative_count'] - weekly_stats['interval_stats'][3]['count'], 'percent': (weekly_stats['negative_count'] - weekly_stats['interval_stats'][3]['count']) / weekly_stats['count'] * 100},
        '震荡': {'lower': -1, 'upper': 1, 'count': weekly_stats['zero_count'] + weekly_stats['interval_stats'][3]['count'] + weekly_stats['interval_stats'][5]['count'], 'percent': (weekly_stats['zero_count'] + weekly_stats['interval_stats'][3]['count'] + weekly_stats['interval_stats'][5]['count']) / weekly_stats['count'] * 100},
        '多头': {'lower': 1, 'upper': float('inf'), 'count': weekly_stats['positive_count'] - weekly_stats['interval_stats'][5]['count'], 'percent': (weekly_stats['positive_count'] - weekly_stats['interval_stats'][5]['count']) / weekly_stats['count'] * 100}
    }
    
    # 计算组合场景统计数据
    scene_stats, _ = calculate_combination_scene_stats(daily_df, weekly_df)
    
    # 构建场景统计字典便于查找
    stats_dict = {row['组合场景']: {'天数': row['天数'], '占比(%)': row['占比(%)']} for _, row in scene_stats.iterrows()}
    
    # 输出各场景定义和特征（包含时间占比统计）
    add_markdown(f"\n### 各组合场景定义及时间占比统计")
    add_markdown(f"| 周线场景 | 日线场景 | 组合场景名称 | 市场特征 | 天数 | 占比(%) |")
    add_markdown(f"|---------|---------|------------|---------|-----|-------|")
    add_markdown(f"| 周线多头 | 日线多头 | 多头太阳 | 强烈上涨趋势，双周期共振向上 | {stats_dict['多头太阳']['天数']} | {stats_dict['多头太阳']['占比(%)']:.2f}% |")
    add_markdown(f"| 周线多头 | 日线震荡 | 多头少阳 | 周线趋势向上，日线整理 | {stats_dict['多头少阳']['天数']} | {stats_dict['多头少阳']['占比(%)']:.2f}% |")
    add_markdown(f"| 周线多头 | 日线空头 | 多头少阴 | 周线趋势向上，日线短期回调 | {stats_dict['多头少阴']['天数']} | {stats_dict['多头少阴']['占比(%)']:.2f}% |")
    add_markdown(f"| 周线震荡 | 日线多头 | 震荡小阳 | 周线整理，日线短期上涨 | {stats_dict['震荡小阳']['天数']} | {stats_dict['震荡小阳']['占比(%)']:.2f}% |")
    add_markdown(f"| 周线震荡 | 日线震荡 | 震荡平衡 | 双周期均处于整理状态 | {stats_dict['震荡平衡']['天数']} | {stats_dict['震荡平衡']['占比(%)']:.2f}% |")
    add_markdown(f"| 周线震荡 | 日线空头 | 震荡小阴 | 周线整理，日线短期下跌 | {stats_dict['震荡小阴']['天数']} | {stats_dict['震荡小阴']['占比(%)']:.2f}% |")
    add_markdown(f"| 周线空头 | 日线多头 | 空头小阳 | 周线趋势向下，日线短期反弹 | {stats_dict['空头小阳']['天数']} | {stats_dict['空头小阳']['占比(%)']:.2f}% |")
    add_markdown(f"| 周线空头 | 日线震荡 | 空头少阴 | 周线趋势向下，日线整理 | {stats_dict['空头少阴']['天数']} | {stats_dict['空头少阴']['占比(%)']:.2f}% |")
    add_markdown(f"| 周线空头 | 日线空头 | 空头太阴 | 强烈下跌趋势，双周期共振向下 | {stats_dict['空头太阴']['天数']} | {stats_dict['空头太阴']['占比(%)']:.2f}% |")
    
    # 输出当前市场的各场景占比
    add_markdown(f"\n### 日线各场景占比")
    add_markdown(f"| 场景 | 斜率区间 | 数据点数量 | 占比(%) |")
    add_markdown(f"|-----|---------|----------|-------|")
    for scene, data in daily_slope_scenes.items():
        range_str = f"< {data['upper']}" if data['lower'] == -float('inf') else f"> {data['lower']}" if data['upper'] == float('inf') else f"{data['lower']} ~ {data['upper']}"
        add_markdown(f"| 日线{scene} | {range_str} | {data['count']} | {data['percent']:.2f}% |")
    
    add_markdown(f"\n### 周线各场景占比")
    add_markdown(f"| 场景 | 斜率区间 | 数据点数量 | 占比(%) |")
    add_markdown(f"|-----|---------|----------|-------|")
    for scene, data in weekly_slope_scenes.items():
        range_str = f"< {data['upper']}" if data['lower'] == -float('inf') else f"> {data['lower']}" if data['upper'] == float('inf') else f"{data['lower']} ~ {data['upper']}"
        add_markdown(f"| 周线{scene} | {range_str} | {data['count']} | {data['percent']:.2f}% |")
    
    # 输出当前市场组合场景
    add_markdown(f"\n### 当前市场组合场景")
    add_markdown(f"基于最近10年数据的平均斜率分析，当前市场整体处于：**{current_combination}** 状态。")
    add_markdown(f"- 日线平均斜率: {daily_avg_slope:.2f}°，属于{daily_overall_scene}市场")
    add_markdown(f"- 周线平均斜率: {weekly_avg_slope:.2f}°，属于{weekly_overall_scene}市场")
    

    
    # 分析组合场景特征
    add_markdown(f"\n### 组合场景特征分析")
    most_common_scene = scene_stats.loc[scene_stats['占比(%)'].idxmax(), '组合场景']
    highest_percentage = scene_stats['占比(%)'].max()
    add_markdown(f"- **最常见场景**: {most_common_scene}，占比 {highest_percentage:.2f}%")
    
    # 分析多头相关场景总占比
    bullish_scenes = ['多头太阳', '多头少阳', '多头少阴']
    bullish_total = scene_stats[scene_stats['组合场景'].isin(bullish_scenes)]['占比(%)'].sum()
    add_markdown(f"- **多头相关场景总占比**: {bullish_total:.2f}%")
    
    # 分析空头相关场景总占比
    bearish_scenes = ['空头太阴', '空头少阴', '空头小阳']
    bearish_total = scene_stats[scene_stats['组合场景'].isin(bearish_scenes)]['占比(%)'].sum()
    add_markdown(f"- **空头相关场景总占比**: {bearish_total:.2f}%")
    
    # 分析震荡相关场景总占比
    sideways_scenes = ['震荡平衡', '震荡小阳', '震荡小阴']
    sideways_total = scene_stats[scene_stats['组合场景'].isin(sideways_scenes)]['占比(%)'].sum()
    add_markdown(f"- **震荡相关场景总占比**: {sideways_total:.2f}%")
    
    # 添加投资建议
    add_markdown(f"\n## 基于斜率分布的投资建议")
    
    # 总体市场判断
    if daily_stats['positive_percent'] > 60 and weekly_stats['positive_percent'] > 60:
        add_markdown(f"- **总体判断**: 长期看涨 - 日线和周线的正斜率占比均超过60%，市场整体处于上升趋势。")
    elif daily_stats['positive_percent'] < 40 and weekly_stats['positive_percent'] < 40:
        add_markdown(f"- **总体判断**: 长期看跌 - 日线和周线的正斜率占比均低于40%，市场整体处于下降趋势。")
    else:
        add_markdown(f"- **总体判断**: 震荡市场 - 日线和周线的斜率分布较为均衡，市场可能处于震荡整理阶段。")
    
    # 震荡市斜率空间建议
    add_markdown(f"\n### 震荡市斜率空间建议")
    add_markdown(f"- **识别特征**: 日线斜率在-1°至1°区间占比约34.58%（轻微下跌17.57%+轻微上涨14.39%+零斜率2.62%），周线斜率在-1°至1°区间占比约16.82%，市场处于震荡整理阶段。")
    add_markdown(f"- **操作建议**: 震荡市可关注斜率在-1°至1°之间的交易机会，适合采用区间交易策略，在斜率接近-1°时考虑低吸，接近1°时考虑高抛。")
    
    # 多头市场斜率空间建议
    add_markdown(f"\n### 多头市场斜率空间建议")
    add_markdown(f"- **识别特征**: 日线斜率在1°以上占比约30.94%（中度上涨22.90%+陡峭上涨4.06%+极陡上涨3.98%），周线斜率在1°以上占比约29.37%（中度上涨11.21%+陡峭上涨10.76%+极陡上涨7.40%）。")
    add_markdown(f"- **操作建议**: 多头市场可关注斜率在1°至3°（中度上涨区间）的持仓机会，此区间占比最高，趋势相对稳定；当斜率超过3°时（陡峭/极陡上涨），需警惕短期过热风险。")
    
    # 空头市场斜率空间建议
    add_markdown(f"\n### 空头市场斜率空间建议")
    add_markdown(f"- **识别特征**: 日线斜率在-1°以下占比约37.09%（中度下跌25.83%+陡峭下跌9.06%+极陡下跌2.20%），周线斜率在-1°以下占比约53.82%（中度下跌19.96%+陡峭下跌15.47%+极陡下跌18.39%）。")
    add_markdown(f"- **操作建议**: 空头市场中，斜率在-3°至-1°（中度下跌区间）占比最高，可作为主要观察区间；当斜率低于-3°时（陡峭/极陡下跌），可能接近超跌区域，注意把握反弹机会。")
    add_markdown(f"- **把握反弹机会**: 当市场处于超跌状态（斜率低于-3°）后，可重点关注以下信号：")
    add_markdown(f"  1. 斜率从极端负值开始回升（从<-3°回升至>-3°但仍为负），表明下跌动能开始减弱")
    add_markdown(f"  2. **日线斜率靠近0°至1°区间**，是重要的反弹信号，说明短期趋势可能从下跌转为走平或小幅上涨")
    add_markdown(f"  3. 结合组合场景分析，若周线仍处于空头但日线出现\"空头小阳\"（周线空头+日线多头）或\"空头少阴\"（周线空头+日线震荡），可考虑轻仓参与反弹")
    add_markdown(f"  4. 反弹操作建议采用小仓位试探策略，设置严格止损，因为整体趋势仍处于空头市场")
    
    # 保存报告
    save_to_markdown(stock_code)

if __name__ == "__main__":
    main()
