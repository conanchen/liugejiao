# 简单查看回测报告内容的脚本

# 打开报告文件并读取内容
try:
    with open(r'd:\git\liugejiao\alltrends\DuoKongsz.000002.md', 'r', encoding='utf-8-sig') as f:
        lines = f.readlines()
    
    print(f"文件共有 {len(lines)} 行\n")
    
    # 打印文件的部分内容来验证
    print("文件前20行内容：")
    for i, line in enumerate(lines[:20]):
        print(f"{i+1}. {line.strip()}")
    
    print(f"\n...省略中间内容...\n")
    
    # 搜索包含'目标仓位'或'震荡市'的行
    print("搜索包含'目标仓位'或'震荡市'的部分行：")
    count = 0
    for line in lines:
        if '目标仓位' in line or '震荡市' in line:
            print(line.strip())
            count += 1
            if count >= 10:  # 最多显示10行
                break
    
    # 搜索场景转换相关的内容
    print("\n搜索场景转换相关内容：")
    scene_lines = [line.strip() for line in lines if '多头太阳' in line or '多头小阴' in line or '空头小阳' in line]
    for line in scene_lines[:15]:  # 最多显示15行
        print(line)
    
    # 检查总收益率
    for line in lines:
        if '总收益率' in line:
            print(f"\n{line.strip()}")
            break
    
except Exception as e:
    print(f"读取文件时出错：{e}")