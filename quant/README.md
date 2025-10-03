# Python量化交易程序

这是一个使用 baostock 获取股票数据的 Python 量化交易程序框架。

## 项目结构
```
alltrends/
├── main.py                  # 主程序文件
├── config.py                # 配置文件
├── requirements.txt         # Python依赖包列表
├── environment.yml          # Conda环境配置
├── conda_source_config.md   # Conda国内源配置指南
├── .gitignore               # Git忽略文件配置
└── README.md                # 项目说明文档
```

## 功能特点
- 使用 baostock API 获取股票和指数的历史日线数据
- 支持前复权、后复权和不复权数据获取
- 计算常用技术指标（如移动平均线）
- 数据可视化（价格走势图、均线图等）
- 数据导出功能（保存为 CSV 文件）

## 依赖安装
本项目已配置为使用Conda环境，因为您确认要使用Conda。

### 关于Conda环境的说明
- Conda环境会自带独立的Python解释器（在environment.yml中指定了python=3.9）
- 所有依赖包都会安装在独立的环境中，不会影响系统其他Python环境
- 使用Conda可以更好地管理科学计算库之间的依赖关系
- 项目已配置使用清华大学的Conda镜像源，以提高国内下载速度

### 创建和激活Conda环境
```bash
# 创建环境（这会自动使用配置文件中指定的清华大学镜像源，提高国内下载速度）
conda env create -f environment.yml

# 激活环境
conda activate alltrends

# 验证环境是否正确激活
python --version  # 应该显示Python 3.9.x版本
conda list        # 应该显示所有已安装的依赖包
```

### 永久配置Conda国内源（可选）
如果您希望所有Conda环境都使用国内源，可以进行全局配置：

1. 打开或创建`.condarc`配置文件：
   - 在Windows系统中，文件路径为：`%USERPROFILE%\.condarc`
   - 您可以使用记事本或其他文本编辑器打开这个文件

2. 将以下内容复制到文件中：
```yaml
channels:
  - defaults
show_channel_urls: true
channel_alias: https://mirrors.tuna.tsinghua.edu.cn/anaconda
default_channels:
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/r
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/pro
  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/msys2
custom_channels:
  conda-forge: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  msys2: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  bioconda: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  menpo: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  pytorch: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
  simpleitk: https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud
```

3. 保存文件并验证配置是否成功：
```bash
conda config --show channels
```

### 临时使用国内源（可选）
如果您只想临时使用国内源来创建环境，可以使用以下命令：

```bash
# 使用清华大学源创建环境
conda env create -f environment.yml -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/
```

### 删除之前的Python虚拟环境（可选）
如果您想删除之前创建的Python虚拟环境，可以执行以下命令：
```bash
# Windows系统
rmdir /s /q venv

# 或手动删除venv文件夹
```

## 使用说明
1. 确保已安装所有依赖包
2. 运行主程序：
   ```bash
   python main.py
   ```
3. 程序会自动获取过去一年的上证指数和贵州茅台（sh.600519）的日线数据
4. 获取的数据会保存在当前目录下的 CSV 文件中
5. 程序会显示价格走势图和均线图

## Conda国内源配置指南
如果您在使用Conda创建环境或安装依赖时遇到下载速度慢的问题，请参考项目中的`conda_source_config.md`文件，其中包含了详细的国内镜像源配置方法和常见问题解决步骤。

**特别提示**：如果您按照指南配置了`.condarc`文件后，执行`conda config --show channels`命令仍然只显示`defaults`，请务必查看`conda_source_config.md`文件中的"配置源后不生效问题"部分，那里提供了详细的故障排除方法。

## 自定义使用
您可以在 main.py 中修改以下参数来自定义数据获取：
- 股票代码：如 `sh.600519`（贵州茅台）、`sz.000002`（万科A）等
- 日期范围：修改 `start_date` 和 `end_date` 来调整获取数据的时间范围
- 复权类型：调整 `adjustflag` 参数（'1'后复权，'2'不复权，'3'前复权）

## 注意事项
- 使用 baostock API 需要网络连接
- 数据获取可能受到 baostock 服务器限制，请合理设置请求频率
- 本程序仅作为学习和研究使用，不构成任何投资建议

## 扩展建议
- 添加更多技术指标计算（MACD、RSI、KDJ等）
- 实现简单的策略回测功能
- 添加多线程/多进程数据获取以提高效率
- 实现定时任务自动获取数据