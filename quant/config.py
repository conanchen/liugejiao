# 配置文件

# 股票代码配置
STOCK_CODES = {
    'SH600519': '贵州茅台',
    # 'SH600036': '招商银行',
    # 'SZ000858': '五粮液',
    'SZ000002': '万科A',
    'SH510300': '沪深300ETF',
    # 可以添加更多股票代码
}

# 指数代码配置
INDEX_CODES = {
    'SH000001': '上证指数',
    'SZ399001': '深证成指',
    'SH000300': '沪深300',
    'SH000016': '上证50',
    # 可以添加更多指数代码
}

# 数据获取配置
DATA_CONFIG = {
    # 日期范围（默认获取过去一年数据）
    'use_relative_dates': False,  # 使用相对日期
    'days_back': 365,  # 从今天往前推的天数
    
    # 如果 use_relative_dates 为 False，则使用下面的固定日期
    'start_date': '2025-09-01',
    'end_date': '2025-09-30',
    
    # 复权类型: '1'后复权, '2'不复权, '3'前复权
    'adjustflag': '3',
    
    # 数据保存路径
    'save_dir': '.',
    
    # 绘制图表配置
    'plot_charts': True,
    'figure_size': (12, 6),
    
    # 移动平均线配置
    'calculate_ma': True,
    'ma_periods': [5, 10, 20, 60],  # 要计算的均线周期
}

# 日志配置
LOGGING_CONFIG = {
    'level': 'INFO',  # 日志级别：DEBUG, INFO, WARNING, ERROR
    'file': 'liugejiao.log',  # 日志文件名
}