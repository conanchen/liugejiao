# 自动化更新脚本

参见 [qlib_updater.py](qlib_updater.py)

## 二、系统服务配置（可选）
### Linux/Mac 定时任务
```bash
# 编辑crontab
crontab -e

# 添加以下行，每天凌晨3点自动更新
0 3 * * * /usr/bin/python3 /path/to/your/qlib_updater.py

# 或者每周一凌晨3点更新
0 3 * * 1 /usr/bin/python3 /path/to/your/qlib_updater.py

```

### Windows 计划任务
创建批处理文件 qlib_update.bat：

```batch
@echo off
python d:\git\liugejiao\qlib\qlib_updater.py
pause
```

然后通过Windows任务计划程序设置每天自动运行。

