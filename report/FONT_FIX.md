# 图表中文显示问题修复指南

## 问题描述

如果生成的图表中中文字符显示为方块（□），这是因为系统缺少中文字体。

## 解决方案

### 方案1：安装中文字体（推荐）

#### Ubuntu/Debian系统
```bash
sudo apt-get update
sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei
# 或者
sudo apt-get install fonts-noto-cjk
```

#### CentOS/RHEL系统
```bash
sudo yum install wqy-microhei-fonts wqy-zenhei-fonts
# 或者
sudo yum install google-noto-cjk-fonts
```

#### 安装后重新生成图表
```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/report
python3 generate_plots_chinese.py
```

### 方案2：使用英文版本

如果无法安装中文字体，可以使用英文版本的脚本：

```bash
cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/report
python3 generate_plots.py
```

这个版本使用英文标签，避免中文显示问题。

### 方案3：手动指定字体路径

如果系统有中文字体但matplotlib找不到，可以手动指定：

```python
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 指定字体路径
font_path = '/usr/share/fonts/truetype/wqy/wqy-microhei.ttc'
prop = fm.FontProperties(fname=font_path)
plt.rcParams['font.family'] = prop.get_name()
```

## 验证字体安装

运行以下命令检查中文字体是否可用：

```python
import matplotlib.font_manager as fm
fonts = [f.name for f in fm.fontManager.ttflist]
chinese_fonts = [f for f in fonts if any(x in f for x in ['WenQuanYi', 'Noto', 'SimHei', 'YaHei'])]
print("可用的中文字体:", chinese_fonts)
```

## 清除matplotlib缓存

如果安装了字体但仍然显示方块，可能需要清除matplotlib缓存：

```bash
rm -rf ~/.cache/matplotlib
python3 generate_plots_chinese.py
```

## 地图名称对照表

| 地图代码 | 中文名称 | 英文拼音 |
|---------|---------|---------|
| ADCC | 暗渡陈仓 | An Du Chen Cang |
| GMZZ | 关门捉贼 | Guan Men Zhuo Zei |
| JCTQ | 金蝉脱壳 | Jin Chan Tuo Ke |
| JDSR | 借刀杀人 | Jie Dao Sha Ren |
| SDJX | 声东击西 | Sheng Dong Ji Xi |
| SWCT | 上屋抽梯 | Shang Wu Chou Ti |
| WWJZ | 围魏救赵 | Wei Wei Jiu Zhao |
| WZSY | 无中生有 | Wu Zhong Sheng You |

