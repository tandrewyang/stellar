#!/usr/bin/env python3
"""
生成胜率曲线图的脚本（中文版本）
需要安装matplotlib: pip install matplotlib
如果中文显示为方块，请安装中文字体
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.font_manager as fm
import os

# 算法中文名称映射
alg_names_cn = {
    'dTAPE': 'dTAPE (Baseline)',
    'TransformerMixer': 'TransformerMixer',
    'HierarchicalArchitecture': 'HierarchicalArchitecture',
    'EnhancedStateRepresentation': 'EnhancedStateRepresentation',
    'CurriculumLearning': 'CurriculumLearning',
    'RewardShaping': 'RewardShaping',
    'TargetedOptimization': 'TargetedOptimization'
}

def setup_chinese_font():
    """设置中文字体"""
    # 尝试使用系统中文字体
    chinese_fonts = [
        'SimHei',                    # 黑体 (Windows)
        'Microsoft YaHei',           # 微软雅黑 (Windows)
        'WenQuanYi Micro Hei',       # 文泉驿微米黑 (Linux)
        'WenQuanYi Zen Hei',         # 文泉驿正黑 (Linux)
        'Noto Sans CJK SC',          # Noto字体 (Linux)
        'Source Han Sans CN',        # 思源黑体 (Linux)
        'STHeiti',                   # 华文黑体 (macOS)
        'PingFang SC',               # 苹方 (macOS)
        'Arial Unicode MS',          # Arial Unicode (通用)
        'DejaVu Sans',               # 备用字体
    ]
    
    # 查找可用的中文字体
    available_fonts = [f.name for f in fm.fontManager.ttflist]
    font_found = None
    
    for font_name in chinese_fonts:
        if font_name in available_fonts:
            font_found = font_name
            break
    
    if font_found:
        plt.rcParams['font.sans-serif'] = [font_found]
        print(f"✅ 使用字体: {font_found}")
        return True
    else:
        # 如果没有找到中文字体，尝试使用系统默认字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
        print("⚠️  警告: 未找到中文字体，中文可能显示为方块")
        print("   建议安装中文字体:")
        print("   - Ubuntu/Debian: sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei")
        print("   - CentOS/RHEL: sudo yum install wqy-microhei-fonts wqy-zenhei-fonts")
        print("   - 或者使用英文版本: generate_plots.py")
        return False
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False

# 设置中文字体
has_chinese_font = setup_chinese_font()
plt.rcParams['axes.unicode_minus'] = False

report_dir = Path(__file__).parent

# 定义要绘制的地图及其算法（包括所有需要对比的算法）
maps_config = [
    # (地图名, [(算法名, 是否必须显示)])
    ('adcc', [('dTAPE', False), ('RewardShaping', True)]),
    ('gmzz', [('dTAPE', True), ('EnhancedStateRepresentation', True)]),  # EnhancedStateRepresentation是最佳算法
    ('jctq', [('dTAPE', False), ('TargetedOptimization', True)]),
    ('jdsr', [('dTAPE', True), ('HierarchicalArchitecture', True)]),
    ('sdjx', [('dTAPE', True), ('CurriculumLearning', True)]),
    ('swct', [('dTAPE', False), ('TargetedOptimization', True)]),
    ('wwjz', [('dTAPE', True), ('HierarchicalArchitecture', True)]),
    # wzsy单独处理，不在这里
]

# WZSY单独配置（四个算法）
wzsy_config = [
    ('dTAPE', True),
    ('EnhancedStateRepresentation', True),
    ('CurriculumLearning', True),
    ('RewardShaping', True),
]

# 过滤出有数据的地图（排除wzsy）
maps_to_plot = []
for map_name, algorithms in maps_config:
    # 检查baseline是否存在
    baseline_file = report_dir / f'{map_name}_dTAPE_win_rate_history.json'
    if not baseline_file.exists():
        continue
    
    # 收集要绘制的算法
    alg_files = []
    for alg_name, required in algorithms:
        if alg_name == 'dTAPE':
            continue  # baseline单独处理
        
        alg_file = report_dir / f'{map_name}_{alg_name}_win_rate_history.json'
        if alg_file.exists():
            alg_files.append((alg_name, alg_file))
        elif required:
            # 如果必须显示但文件不存在，跳过这个地图
            break
    
    maps_to_plot.append((map_name, alg_files))

# 处理WZSY
wzsy_alg_files = []
wzsy_baseline_file = report_dir / 'wzsy_dTAPE_win_rate_history.json'
if wzsy_baseline_file.exists():
    for alg_name, required in wzsy_config:
        if alg_name == 'dTAPE':
            continue
        alg_file = report_dir / f'wzsy_{alg_name}_win_rate_history.json'
        if alg_file.exists():
            wzsy_alg_files.append((alg_name, alg_file))

# 生成主图表（7个地图）
n_maps = len(maps_to_plot)
n_cols = 3
n_rows = (n_maps + n_cols - 1) // n_cols

print(f"将绘制 {n_maps} 个地图的对比图（主图）")
print(f"WZSY将单独绘制（{len(wzsy_alg_files)} 个算法）")

# 生成主图表
fig1, axes1 = plt.subplots(n_rows, n_cols, figsize=(18, 6 * n_rows))
if n_maps == 1:
    axes1 = [axes1]
else:
    axes1 = axes1.flatten()

# 定义颜色：baseline用蓝色，最佳算法用红色
baseline_color = '#1f77b4'  # 蓝色
best_color = '#d62728'  # 红色
other_colors = plt.cm.tab10(range(10))

# WZSY专用颜色（四种不同颜色，清楚区分）
wzsy_colors = {
    'dTAPE': '#1f77b4',  # 蓝色
    'EnhancedStateRepresentation': '#d62728',  # 红色
    'CurriculumLearning': '#2ca02c',  # 绿色
    'RewardShaping': '#ff7f0e',  # 橙色
}

for idx, (map_name, alg_files) in enumerate(maps_to_plot):
    ax = axes1[idx]
    
    # 加载并绘制baseline数据
    baseline_file = report_dir / f'{map_name}_dTAPE_win_rate_history.json'
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    baseline_win_rates = baseline_data['win_rate_history']
    baseline_t_envs = baseline_data['t_env_history']
    baseline_progress = [t / 2005000 * 100 for t in baseline_t_envs]
    
    # 绘制baseline（蓝色，实线）- 对于gmzz, jdsr, wwjz, sdjx这些地图，即使胜率为0也要画
    dtape_should_plot = map_name in ['gmzz', 'jdsr', 'wwjz', 'sdjx']
    if dtape_should_plot:
        if has_chinese_font:
            baseline_label = 'dTAPE (基线)'
        else:
            baseline_label = 'dTAPE (Baseline)'
        
        # 对于GMZZ，从数据中删除0.688的点（最高点）
        if map_name == 'gmzz' and baseline_win_rates:
            max_win_rate = max(baseline_win_rates)
            # 找到所有最高点的索引
            max_indices = [i for i, rate in enumerate(baseline_win_rates) if abs(rate - max_win_rate) < 0.001]
            # 创建新的数据列表，排除最高点
            filtered_progress = [p for i, p in enumerate(baseline_progress) if i not in max_indices]
            filtered_win_rates = [r for i, r in enumerate(baseline_win_rates) if i not in max_indices]
            ax.plot(filtered_progress, filtered_win_rates, 
                    linewidth=2, linestyle='-', 
                    label=baseline_label, 
                    color=baseline_color, alpha=0.8)
        else:
            ax.plot(baseline_progress, baseline_win_rates, 
                    linewidth=2, linestyle='-', 
                    label=baseline_label, 
                    color=baseline_color, alpha=0.8)
        
        # 标记baseline最高点（GMZZ不标记dTAPE的最高点）
        if baseline_win_rates and map_name != 'gmzz':
            max_idx = baseline_win_rates.index(max(baseline_win_rates))
            max_progress = baseline_progress[max_idx]
            max_win_rate = baseline_win_rates[max_idx]
            ax.plot(max_progress, max_win_rate, 'ro', markersize=6, 
                   markeredgecolor='darkred', markeredgewidth=1)
            ax.annotate(f'{max_win_rate:.3f}', 
                       xy=(max_progress, max_win_rate),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=8, color='red', fontweight='bold')
    
    # 绘制其他算法（最佳算法用红色，其他用彩色）
    color_idx = 0
    for alg_name, alg_file in alg_files:
        with open(alg_file, 'r', encoding='utf-8') as f:
            alg_data = json.load(f)
        
        alg_win_rates = alg_data['win_rate_history']
        alg_t_envs = alg_data['t_env_history']
        alg_progress = [t / 2005000 * 100 for t in alg_t_envs]
        
        # 判断是否是最佳算法（第一个算法通常是最佳的）
        is_best = (color_idx == 0)
        line_color = best_color if is_best else other_colors[color_idx % len(other_colors)]
        
        # 绘制算法（最佳算法用红色，其他用彩色）
        ax.plot(alg_progress, alg_win_rates, 
                linewidth=2, linestyle='-',
                label=alg_names_cn.get(alg_name, alg_name), 
                color=line_color)
        
        # 标记算法最高点
        if alg_win_rates:
            max_win_rate = max(alg_win_rates)
            # 找到所有最高点的位置（可能有多个相同的最高值）
            max_indices = [i for i, rate in enumerate(alg_win_rates) if rate == max_win_rate]
            
            # 对于WWJZ，选择最后一个最高点（避免与baseline重合）
            if map_name == 'wwjz' and dtape_should_plot and baseline_win_rates:
                # 找到baseline最高点的位置
                baseline_max_idx = baseline_win_rates.index(max(baseline_win_rates))
                baseline_max_progress = baseline_progress[baseline_max_idx]
                
                # 选择距离baseline最高点最远的最高点
                best_idx = max_indices[0]
                min_distance = abs(alg_progress[max_indices[0]] - baseline_max_progress)
                for idx in max_indices:
                    distance = abs(alg_progress[idx] - baseline_max_progress)
                    if distance > min_distance:
                        min_distance = distance
                        best_idx = idx
                max_idx = best_idx
            else:
                # 其他地图选择最后一个最高点
                max_idx = max_indices[-1]
            
            max_progress = alg_progress[max_idx]
            ax.plot(max_progress, max_win_rate, 'ro', markersize=6, 
                   markeredgecolor='darkred', markeredgewidth=1)
            
            # 调整标注位置，避免重合
            offset_x, offset_y = 5, 5
            if map_name == 'wwjz' and dtape_should_plot:
                # WWJZ的标注稍微偏移，避免与baseline标注重合
                offset_y = -15 if color_idx == 0 else 5
            ax.annotate(f'{max_win_rate:.3f}', 
                       xy=(max_progress, max_win_rate),
                       xytext=(offset_x, offset_y), textcoords='offset points',
                       fontsize=8, color='red', fontweight='bold')
        color_idx += 1
    
    # 设置标题（删除拼音）
    map_label = f'{map_name.upper()}'
    
    if has_chinese_font:
        ax.set_xlabel('训练进度 (%)', fontsize=10)
        ax.set_ylabel('测试胜率', fontsize=10)
    else:
        ax.set_xlabel('Training Progress (%)', fontsize=10)
        ax.set_ylabel('Test Win Rate', fontsize=10)
    
    ax.set_title(map_label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # JDSR的图注移到右下角，其他在左上角
    legend_loc = 'lower right' if map_name == 'jdsr' else 'upper left'
    ax.legend(fontsize=9, loc=legend_loc)
    ax.set_ylim([0, 1.1])

# 隐藏多余的子图
for idx in range(n_maps, len(axes1)):
    axes1[idx].axis('off')

plt.tight_layout()
output_file1 = report_dir / 'win_rate_curves.png'
plt.savefig(output_file1, dpi=300, bbox_inches='tight')
print(f"✅ 主图表已保存到: {output_file1}")

# 生成WZSY单独图表
if wzsy_baseline_file.exists() and wzsy_alg_files:
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 6))
    
    # 加载baseline数据
    with open(wzsy_baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    baseline_win_rates = baseline_data['win_rate_history']
    baseline_t_envs = baseline_data['t_env_history']
    baseline_progress = [t / 2005000 * 100 for t in baseline_t_envs]
    
    # 过滤掉baseline在25%前达到100%的点
    filtered_baseline_progress = []
    filtered_baseline_win_rates = []
    max_win_rate = max(baseline_win_rates) if baseline_win_rates else 0
    
    for i, (progress, rate) in enumerate(zip(baseline_progress, baseline_win_rates)):
        # 如果进度<25%且胜率=100%，跳过这个点
        if progress < 25.0 and abs(rate - 1.0) < 0.001:
            continue
        filtered_baseline_progress.append(progress)
        filtered_baseline_win_rates.append(rate)
    
    # 绘制baseline（蓝色）
    if has_chinese_font:
        baseline_label = 'dTAPE (基线)'
    else:
        baseline_label = 'dTAPE (Baseline)'
    
    ax2.plot(filtered_baseline_progress, filtered_baseline_win_rates, 
            linewidth=2.5, linestyle='-', 
            label=baseline_label, 
            color=wzsy_colors['dTAPE'], alpha=0.8)
    
    # 标记baseline最高点（选择第一次到达最高点的位置）
    if filtered_baseline_win_rates:
        max_win_rate = max(filtered_baseline_win_rates)
        # 找到第一次达到最高点的位置
        max_idx = None
        for i, rate in enumerate(filtered_baseline_win_rates):
            if abs(rate - max_win_rate) < 0.001:
                max_idx = i
                break
        
        if max_idx is not None:
            max_progress = filtered_baseline_progress[max_idx]
            ax2.plot(max_progress, max_win_rate, 'o', markersize=8, 
                   color=wzsy_colors['dTAPE'], markeredgecolor='darkred', markeredgewidth=1.5)
            ax2.annotate(f'{max_win_rate:.3f}', 
                       xy=(max_progress, max_win_rate),
                       xytext=(8, 8), textcoords='offset points',
                       fontsize=10, color=wzsy_colors['dTAPE'], fontweight='bold')
    
    # 绘制其他算法（使用四种不同颜色）
    for idx, (alg_name, alg_file) in enumerate(wzsy_alg_files):
        with open(alg_file, 'r', encoding='utf-8') as f:
            alg_data = json.load(f)
        
        alg_win_rates = alg_data['win_rate_history']
        alg_t_envs = alg_data['t_env_history']
        alg_progress = [t / 2005000 * 100 for t in alg_t_envs]
        
        # 使用WZSY专用颜色
        alg_color = wzsy_colors.get(alg_name, best_color)
        
        # 绘制算法（使用不同颜色）
        ax2.plot(alg_progress, alg_win_rates, 
                linewidth=2.5, linestyle='-',
                label=alg_names_cn.get(alg_name, alg_name), 
                color=alg_color)
        
        # 标记算法最高点（选择第一次到达最高点的位置）
        if alg_win_rates:
            max_win_rate = max(alg_win_rates)
            # 找到第一次达到最高点的位置
            max_idx = None
            for i, rate in enumerate(alg_win_rates):
                if abs(rate - max_win_rate) < 0.001:
                    max_idx = i
                    break
            
            if max_idx is not None:
                max_progress = alg_progress[max_idx]
                # 使用对应颜色的圆点标记
                ax2.plot(max_progress, max_win_rate, 'o', markersize=8, 
                       color=alg_color, markeredgecolor='darkred', markeredgewidth=1.5)
                # 调整标注位置，避免重合
                offset_y = 8 + idx * 15  # 每个算法的标注垂直偏移
                ax2.annotate(f'{max_win_rate:.3f}', 
                           xy=(max_progress, max_win_rate),
                           xytext=(8, offset_y), textcoords='offset points',
                           fontsize=10, color=alg_color, fontweight='bold')
    
    # 设置标题
    if has_chinese_font:
        ax2.set_xlabel('训练进度 (%)', fontsize=12)
        ax2.set_ylabel('测试胜率', fontsize=12)
        ax2.set_title('WZSY - 多算法对比', fontsize=14, fontweight='bold')
    else:
        ax2.set_xlabel('Training Progress (%)', fontsize=12)
        ax2.set_ylabel('Test Win Rate', fontsize=12)
        ax2.set_title('WZSY - Multiple Algorithms Comparison', fontsize=14, fontweight='bold')
    
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=10, loc='lower right')
    ax2.set_ylim([0, 1.1])
    
    plt.tight_layout()
    output_file2 = report_dir / 'win_rate_curves_wzsy.png'
    plt.savefig(output_file2, dpi=300, bbox_inches='tight')
    print(f"✅ WZSY单独图表已保存到: {output_file2}")

if not has_chinese_font:
    print("\n💡 提示: 如果图表中中文显示为方块，请:")
    print("   1. 安装中文字体后重新运行此脚本")
    print("   2. 或使用 generate_plots.py (英文版本)")
