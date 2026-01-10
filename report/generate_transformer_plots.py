#!/usr/bin/env python3
"""
生成TransformerMixer与Baseline对比图的脚本
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

# 定义要绘制的地图（TransformerMixer表现最好的地图）
# WZSY最高胜率达到100%，ADCC最高胜率96.88%
maps_config = [
    ('wzsy', 'WZSY'),  # TransformerMixer最高胜率100%
    ('adcc', 'ADCC'),  # TransformerMixer最高胜率96.88%
]

# 定义颜色：baseline用蓝色，TransformerMixer用红色
baseline_color = '#1f77b4'  # 蓝色
transformer_color = '#d62728'  # 红色

# 生成图表
n_maps = len(maps_config)
n_cols = 2
n_rows = 1

print(f"将绘制 {n_maps} 个地图的TransformerMixer vs Baseline对比图")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 6))
if n_maps == 1:
    axes = [axes]
else:
    axes = axes.flatten()

for idx, (map_name, map_label) in enumerate(maps_config):
    ax = axes[idx]
    
    # 加载baseline数据
    baseline_file = report_dir / f'{map_name}_dTAPE_win_rate_history.json'
    if not baseline_file.exists():
        print(f"⚠️  警告: {baseline_file} 不存在，跳过 {map_name}")
        ax.axis('off')
        continue
    
    with open(baseline_file, 'r', encoding='utf-8') as f:
        baseline_data = json.load(f)
    
    baseline_win_rates = baseline_data['win_rate_history']
    baseline_t_envs = baseline_data['t_env_history']
    baseline_progress = [t / 2005000 * 100 for t in baseline_t_envs]
    
    # 绘制baseline（蓝色，实线）
    ax.plot(baseline_progress, baseline_win_rates, 
            linewidth=2, linestyle='-', 
            label='dTAPE (Baseline)', 
            color=baseline_color, alpha=0.8)
    
    # 标记baseline最高点
    if baseline_win_rates:
        max_idx = baseline_win_rates.index(max(baseline_win_rates))
        max_progress = baseline_progress[max_idx]
        max_win_rate = baseline_win_rates[max_idx]
        ax.plot(max_progress, max_win_rate, 'bo', markersize=6, 
               markeredgecolor='darkblue', markeredgewidth=1)
        ax.annotate(f'{max_win_rate:.3f}', 
                   xy=(max_progress, max_win_rate),
                   xytext=(5, 5), textcoords='offset points',
                   fontsize=8, color='blue', fontweight='bold')
    
    # 加载TransformerMixer数据
    transformer_file = report_dir / f'{map_name}_TransformerMixer_win_rate_history.json'
    if not transformer_file.exists():
        print(f"⚠️  警告: {transformer_file} 不存在，跳过 {map_name}")
        continue
    
    with open(transformer_file, 'r', encoding='utf-8') as f:
        transformer_data = json.load(f)
    
    transformer_win_rates = transformer_data['win_rate_history']
    transformer_t_envs = transformer_data.get('t_env_history', [])
    
    # 如果没有t_env_history，使用索引作为进度
    if transformer_t_envs:
        transformer_progress = [t / 2005000 * 100 for t in transformer_t_envs]
    else:
        # 假设数据点均匀分布
        max_t_env = 2005000
        transformer_progress = [i * max_t_env / len(transformer_win_rates) / max_t_env * 100 
                               for i in range(len(transformer_win_rates))]
    
    # 绘制TransformerMixer（红色，实线）
    ax.plot(transformer_progress, transformer_win_rates, 
            linewidth=2, linestyle='-',
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
        ax.plot(max_progress, max_win_rate, 'ro', markersize=6, 
               markeredgecolor='darkred', markeredgewidth=1)
        
        # 调整标注位置，避免与baseline重合
        offset_y = -15 if baseline_win_rates and abs(baseline_progress[baseline_win_rates.index(max(baseline_win_rates))] - max_progress) < 5 else 5
        ax.annotate(f'{max_win_rate:.3f}', 
                   xy=(max_progress, max_win_rate),
                   xytext=(5, offset_y), textcoords='offset points',
                   fontsize=8, color='red', fontweight='bold')
    
    # 设置标题和标签
    ax.set_xlabel('Training Progress (%)', fontsize=10)
    ax.set_ylabel('Test Win Rate', fontsize=10)
    ax.set_title(map_label, fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='upper left')
    ax.set_ylim([0, 1.1])

plt.tight_layout()
output_file = report_dir / 'win_rate_curves_transformer.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"✅ TransformerMixer对比图表已保存到: {output_file}")

