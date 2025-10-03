#!/usr/bin/env python3
"""
Qlib数据自动更新脚本
每天自动检查并下载最新的社区数据源
"""

import os
import requests
import tarfile
import datetime
import json
import pandas as pd
from pathlib import Path
import logging

# Qlib相关导入
import qlib
from qlib.constant import REG_CN
from qlib.data import D

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('qlib_updater.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class Qlib数据自动更新器:
    def __init__(self, 数据目录="~/qlib_data"):
        self.数据目录 = Path(数据目录).expanduser()
        self.版本文件 = self.数据目录 / "current_version.json"
        self.社区数据源 = "https://github.com/chenditc/investment_data/releases/latest/download/qlib_bin.tar.gz"
        
        # 创建数据目录
        self.数据目录.mkdir(parents=True, exist_ok=True)
        
    def 获取当前版本(self):
        """获取当前已安装的数据版本"""
        if self.版本文件.exists():
            with open(self.版本文件, 'r', encoding='utf-8') as f:
                return json.load(f)
        return {"version": "unknown", "last_updated": None}
    
    def 保存当前版本(self, 版本信息):
        """保存当前版本信息"""
        with open(self.版本文件, 'w', encoding='utf-8') as f:
            json.dump(版本信息, f, ensure_ascii=False, indent=2)
    
    def 检查最新版本(self):
        """检查社区数据源的最新版本"""
        try:
            # 获取最新的发布信息
            releases_url = "https://api.github.com/repos/chenditc/investment_data/releases/latest"
            response = requests.get(releases_url, timeout=10)
            response.raise_for_status()
            
            release_info = response.json()
            发布日期 = release_info['published_at'][:10]  # 提取YYYY-MM-DD
            下载链接 = release_info['assets'][0]['browser_download_url']
            
            logging.info(f"发现最新版本: {发布日期}, 下载链接: {下载链接}")
            return 发布日期, 下载链接
            
        except Exception as e:
            logging.error(f"检查最新版本失败: {e}")
            return None, None
    
    def 下载数据(self, 下载链接):
        """下载数据包"""
        try:
            # 临时文件路径
            临时文件 = self.数据目录 / "qlib_bin_temp.tar.gz"
            
            logging.info("开始下载数据包...")
            response = requests.get(下载链接, stream=True, timeout=60)
            response.raise_for_status()
            
            # 下载文件
            with open(临时文件, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            logging.info("下载完成")
            return 临时文件
            
        except Exception as e:
            logging.error(f"下载数据失败: {e}")
            return None
    
    def 解压数据(self, 压缩文件路径):
        """解压数据包"""
        try:
            logging.info("开始解压数据...")
            
            # 备份旧数据（如果存在）
            备份目录 = self.数据目录 / "backup"
            备份目录.mkdir(exist_ok=True)
            
            if (self.数据目录 / "qlib_bin" / "calendars").exists():
                备份时间 = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                (self.数据目录 / "qlib_bin" / "calendars").rename(备份目录 / f"calendars_{备份时间}")
                (self.数据目录 / "qlib_bin" / "features").rename(备份目录 / f"features_{备份时间}")
                (self.数据目录 / "qlib_bin" / "instruments").rename(备份目录 / f"instruments_{备份时间}")
            
            # 解压新数据
            with tarfile.open(压缩文件路径, 'r:gz') as tar:
                tar.extractall(self.数据目录)
            

            # 删除临时文件
            压缩文件路径.unlink()
            
            logging.info("解压完成")
            return True
            
        except Exception as e:
            logging.error(f"解压数据失败: {e}")
            return False
    
    def 验证数据完整性(self):
        """验证数据是否完整可用"""
        try:
            # 检查必要的目录和文件
            必要目录 = ["qlib_bin/calendars", "qlib_bin/features", "qlib_bin/instruments"]
            for 目录 in 必要目录:
                if not (self.数据目录 / 目录).exists():
                    logging.error(f"缺少必要目录: {目录}")
                    return False
            
            # 尝试初始化Qlib来验证数据
            qlib.init(provider_uri=str(self.数据目录 / "qlib_bin"), region=REG_CN)
            
            # 测试获取数据 - 获取用户需要的字段
            test_data = D.features(
                instruments=["SZ000030"],
                fields=["$open", "$high", "$low", "$close", "$volume"],
                start_time="2024-01-01",
                end_time="2024-01-10"
            )
            
            if test_data.empty:
                logging.warning("测试数据获取为空，但目录结构正常")
            else:
                logging.info("数据验证成功")
                # 显示数据基本信息
                logging.info(f"数据类型: {type(test_data)}")
                logging.info(f"数据形状: {test_data.shape}")
                logging.info(f"数据索引: {test_data.index.names}")
                logging.info(f"数据列: {list(test_data.columns)}")
                # 显示数据字段类型
                logging.info(f"数据字段类型:\n{test_data.dtypes}")
                # 只显示前3行数据
                logging.info(f"测试数据样例（前3行）:\n{test_data.head(3)}")
                # Qlib数据说明
                logging.info("注：Qlib中的数据通常是经过预处理的标准化数据，数值可能是浮点数格式。")
                logging.info("成交量等字段可能经过了单位转换或标准化处理。")
            
            return True
            
        except Exception as e:
            logging.error(f"数据验证失败: {e}")
            return False
    
    def 执行更新(self):
        """执行完整的更新流程"""
        logging.info("开始检查Qlib数据更新...")
        
        # 获取当前版本
        当前版本 = self.获取当前版本()
        logging.info(f"当前数据版本: {当前版本.get('version', '未知')}")
        
        # 检查最新版本
        最新版本, 下载链接 = self.检查最新版本()
        if not 最新版本:
            logging.error("无法获取最新版本信息")
            return False
        
        # 检查是否需要更新
        if 当前版本.get('version') == 最新版本:
            logging.info("当前已是最新版本，无需更新")
            return True
        
        logging.info(f"发现新版本: {最新版本}, 开始更新...")
        
        # 下载数据
        压缩文件 = self.下载数据(下载链接)
        if not 压缩文件:
            return False
        
        # 解压数据
        if not self.解压数据(压缩文件):
            return False
        
        # 验证数据
        if not self.验证数据完整性():
            logging.error("数据验证失败，更新未完成")
            return False
        
        # 更新版本信息
        新版本信息 = {
            "version": 最新版本,
            "last_updated": datetime.datetime.now().isoformat(),
            "data_source": "community"
        }
        self.保存当前版本(新版本信息)
        
        logging.info(f"Qlib数据更新完成! 新版本: {最新版本}")
        return True

def 初始化Qlib数据(数据目录="~/qlib_data"):
    """
    初始化Qlib数据（如果不存在则下载）
    """
    更新器 = Qlib数据自动更新器(数据目录)
    
    logging.info(f"Qlib数据目录: {更新器.数据目录}")

    # 检查数据是否存在
    if not (更新器.数据目录 / "calendars").exists():
        logging.info("未发现现有数据，开始首次下载...")
        return 更新器.执行更新()
    else:
        logging.info("数据目录已存在，跳过首次下载")
        return True

if __name__ == "__main__":
    # 直接运行脚本时执行更新
    更新器 = Qlib数据自动更新器("D:/data")
    结果 = 更新器.执行更新()
    
    # 强制验证数据完整性以显示测试数据
    if 结果:
        logging.info("强制执行数据完整性验证以显示测试数据...")
        更新器.验证数据完整性()