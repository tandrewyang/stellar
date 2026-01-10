#!/usr/bin/env python3
"""
生成TransformerMixer (WZSY达到100%)与Baseline对比图的脚本
模仿win_rate_curves.png的格式
"""
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

# 算法名称映射
alg_names = {
    'dTAPE': 'dTAPE (Baseline)',
    'TransformerMixer': 'TransformerMixer',
}

report_dir = Path(__file__).parent

# 支持的地图（可以通过命令行参数选择）
import sys
if len(sys.argv) > 1:
    map_name = sys.argv[1].lower()
    if map_name not in ['wzsy', 'jdsr', 'adcc']:
        print(f"错误: 不支持的地图 '{map_name}'")
        print("支持的地图: wzsy, jdsr, adcc")
        sys.exit(1)
else:
    map_name = 'wzsy'  # 默认

map_labels = {
    'wzsy': 'WZSY',
    'jdsr': 'JDSR',
    'adcc': 'ADCC'
}
map_label = map_labels[map_name]

# 定义颜色：baseline用蓝色，TransformerMixer用红色
baseline_color = '#1f77b4'  # 蓝色
transformer_color = '#d62728'  # 红色

print(f"将绘制 {map_label} 地图的TransformerMixer vs Baseline对比图")

# 创建单个子图
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

# 加载baseline数据
baseline_file = report_dir / f'{map_name}_dTAPE_win_rate_history.json'
if not baseline_file.exists():
    print(f"⚠️  警告: {baseline_file} 不存在")
else:
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    baseline_win_rates = baseline_data['win_rate_history']
    baseline_t_envs = baseline_data['t_env_history']
    baseline_progress = [t / 2005000 * 100 for t in baseline_t_envs]
    
    # 绘制baseline（蓝色，实线）
    ax.plot(baseline_progress, baseline_win_rates, 
            linewidth=2.5, linestyle='-', 
            label='dTAPE (Baseline)', 
            color=baseline_color, alpha=0.8)
    
    # 标记baseline最高点
    if baseline_win_rates:
        max_win_rate = max(baseline_win_rates)
        max_indices = [i for i, rate in enumerate(baseline_win_rates) if abs(rate - max_win_rate) < 0.001]
        max_idx = max_indices[-1]  # 选择最后一个最高点
        max_progress = baseline_progress[max_idx]
        ax.plot(max_progress, max_win_rate, 'o', markersize=8, 
               color='red', markeredgecolor='darkred', markeredgewidth=1.5)
        ax.annotate(f'{max_win_rate:.4f}', 
                   xy=(max_progress, max_win_rate),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=10, color='red', fontweight='bold')

# 加载TransformerMixer数据
transformer_file = report_dir / f'{map_name}_TransformerMixer_win_rate_history.json'
if not transformer_file.exists():
    print(f"⚠️  警告: {transformer_file} 不存在")
else:
    with open(transformer_file, 'r', encoding='utf-8') as f:
        transformer_data = json.load(f)
    
    transformer_win_rates = transformer_data['win_rate_history']
    transformer_t_envs = transformer_data.get('t_env_history', [])
    
    # 计算进度，确保长度一致
    if transformer_t_envs:
        # 如果t_env_history长度是win_rate_history的两倍（或接近），每两个取一个
        if len(transformer_t_envs) >= 2 * len(transformer_win_rates):
            # 每两个取一个（取偶数索引），然后截取到正确长度
            transformer_t_envs = transformer_t_envs[::2][:len(transformer_win_rates)]
        # 如果长度不匹配，截取到相同长度
        elif len(transformer_t_envs) > len(transformer_win_rates):
            transformer_t_envs = transformer_t_envs[:len(transformer_win_rates)]
        elif len(transformer_t_envs) < len(transformer_win_rates):
            # 如果t_env_history太短，用最后一个值填充
            last_t_env = transformer_t_envs[-1] if transformer_t_envs else 0
            transformer_t_envs = transformer_t_envs + [last_t_env] * (len(transformer_win_rates) - len(transformer_t_envs))
        
        # 最终确保长度完全一致
        min_len = min(len(transformer_t_envs), len(transformer_win_rates))
        transformer_t_envs = transformer_t_envs[:min_len]
        transformer_win_rates = transformer_win_rates[:min_len]
        
        transformer_progress = [t / 2005000 * 100 for t in transformer_t_envs]
    else:
        # 假设数据点均匀分布
        max_t_env = 2005000 
        transformer_progress = [i * max_t_env / len(transformer_win_rates) / max_t_env * 100 
                               for i in range(len(transformer_win_rates))]
    
    # 绘制TransformerMixer（红色，实线）
    ax.plot(transformer_progress, transformer_win_rates, 
            linewidth=2.5, linestyle='-',
            label='TransformerMixer', 
            color=transformer_color)
    
    # 标记TransformerMixer最高点
    if transformer_win_rates:
        max_win_rate = max(transformer_win_rates)
        # 找到所有最高点的位置（可能有多个相同的最高值）
        max_indices = [i for i, rate in enumerate(transformer_win_rates) if abs(rate - max_win_rate) < 0.001]
        # 选择最后一个最高点
        max_idx = max_indices[-1]
        max_progress = transformer_progress[max_idx]
        ax.plot(max_progress, max_win_rate, 'o', markersize=8, 
               color='red', markeredgecolor='darkred', markeredgewidth=1.5)
        
        # 调整标注位置，避免与baseline重合
        if baseline_win_rates:
            baseline_max_idx = baseline_win_rates.index(max(baseline_win_rates))
            baseline_max_progress = baseline_progress[baseline_max_idx]
            if abs(baseline_max_progress - max_progress) < 5:
                offset_y = -20
            else:
                offset_y = 5
        else:
            offset_y = 5
            
        ax.annotate(f'{max_win_rate:.4f}', 
                   xy=(max_progress, max_win_rate),
                   xytext=(5, offset_y), textcoords='offset points',
                   fontsize=10, color='red', fontweight='bold')

# 设置标题和标签
ax.set_xlabel('Training Progress (%)', fontsize=12)
ax.set_ylabel('Test Win Rate', fontsize=12)
ax.set_title(map_label, fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=11, loc='upper left')
ax.set_ylim([0, 1.1])

plt.tight_layout()
output_file = report_dir / f'win_rate_curves_transformer_{map_name}.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ TransformerMixer对比图表已保存到: {output_file}")
