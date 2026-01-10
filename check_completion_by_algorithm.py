#!/usr/bin/env python3
"""
按算法整理地图完成情况
检查每个算法的results文件夹，包括所有配置目录的实验
"""
import os
import json
from collections import defaultdict
from pathlib import Path

# 算法列表
ALGORITHMS = {
    'CurriculumLearning': {
        'results_dir': 'results'
    },
    'RewardShaping': {
        'results_dir': 'results'
    },
    'dTAPE': {
        'results_dir': 'results'
    },
    'EnhancedStateRepresentation': {
        'results_dir': 'results'
    },
    'HierarchicalArchitecture': {
        'results_dir': 'results'
    },
    'TransformerMixer': {
        'results_dir': 'results'
    }
}

# 所有地图列表（12个地图，都是4个字母）
ALL_MAPS = ['adcc', 'dhls', 'fkwz', 'gmzz', 'jctq', 'jdsr', 'sdjx', 'swct', 'tlhz', 'wwjz', 'wzsy', 'yqgz']

BASE_PATH = Path('/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs')
T_MAX_TARGET = 2005000  # 目标训练步数
COMPLETION_THRESHOLD = 0.95  # 完成阈值（95%）

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None

def get_win_rate(info):
    """从info.json中提取胜率"""
    test_win_rate = info.get('test_battle_won_mean')
    if test_win_rate is not None:
        if isinstance(test_win_rate, list) and len(test_win_rate) > 0:
            val = test_win_rate[-1]
            if isinstance(val, (int, float)):
                return float(val)
        elif isinstance(test_win_rate, (int, float)):
            return float(test_win_rate)
    
    train_win_rate = info.get('battle_won_mean')
    if train_win_rate is not None:
        if isinstance(train_win_rate, list) and len(train_win_rate) > 0:
            val = train_win_rate[-1]
            if isinstance(val, (int, float)):
                return float(val)
        elif isinstance(train_win_rate, (int, float)):
            return float(train_win_rate)
    
    return 0.0

def get_reward(info):
    """从info.json中提取奖励"""
    test_return = info.get('test_return_mean')
    if test_return is not None:
        if isinstance(test_return, list) and len(test_return) > 0:
            val = test_return[-1]
            if isinstance(val, (int, float)):
                return float(val)
        elif isinstance(test_return, (int, float)):
            return float(test_return)
    
    train_return = info.get('return_mean')
    if train_return is not None:
        if isinstance(train_return, list) and len(train_return) > 0:
            val = train_return[-1]
            if isinstance(val, (int, float)):
                return float(val)
        elif isinstance(train_return, (int, float)):
            return float(train_return)
    
    return 0.0

def get_t_env_from_log(log_path):
    """从训练日志中提取最新的t_env值"""
    if not log_path.exists():
        return None
    
    try:
        import re
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in reversed(lines):
                if 't_env:' in line:
                    match = re.search(r't_env:\s*(\d+)', line)
                    if match:
                        return int(match.group(1))
    except Exception as e:
        pass
    
    return None

def check_model_files(alg_path, map_name):
    """检查是否有保存的模型文件"""
    models_dir = alg_path / 'results' / 'models'
    if not models_dir.exists():
        return False
    
    # 查找包含地图名称的模型文件或目录
    for item in models_dir.iterdir():
        if map_name in item.name.lower():
            # 检查是否有.pt或.pth文件
            if item.is_file() and (item.suffix in ['.pt', '.pth']):
                return True
            elif item.is_dir():
                # 检查目录中是否有模型文件
                if any(f.suffix in ['.pt', '.pth'] for f in item.iterdir() if f.is_file()):
                    return True
    
    return False

def find_best_experiment(alg_path, map_name):
    """找到某个地图的最佳实验（检查所有配置目录）"""
    results_dir = alg_path / 'results'
    sacred_dir = results_dir / 'sacred' / map_name
    
    if not sacred_dir.exists():
        return None
    
    best_exp = None
    best_completion = -1
    best_win_rate = -1
    
    # 遍历所有配置目录（不再限制特定配置名称）
    for config_dir in sacred_dir.iterdir():
        if not config_dir.is_dir():
            continue
        
        # 遍历所有实验ID
        for exp_dir in config_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            try:
                int(exp_dir.name)
            except ValueError:
                continue
            
            info_path = exp_dir / 'info.json'
            config_path = exp_dir / 'config.json'
            
            if not info_path.exists():
                continue
            
            info = load_json(info_path)
            if not info:
                continue
            
            # 从config.json获取t_max
            t_max = T_MAX_TARGET
            if config_path.exists():
                config = load_json(config_path)
                if config:
                    t_max = config.get('t_max', T_MAX_TARGET)
            
            if t_max < 100000:
                continue
            
            # 获取t_env - 优先使用info.json中的值
            t_env = info.get('t_env', 0)
            
            # 如果info.json中没有t_env，从episode估算（更准确，因为每个实验的episode不同）
            if t_env == 0:
                episodes = info.get('episode', [])
                if episodes and len(episodes) > 0:
                    last_episode = episodes[-1] if isinstance(episodes, list) else episodes
                    t_env = last_episode * 200
            
            # 如果还是没有，尝试从训练日志中获取（作为最后手段，但可能不准确）
            if t_env == 0:
                train_logs_dir = results_dir / 'train_logs'
                if train_logs_dir.exists():
                    # 尝试找到与实验ID相关的日志文件
                    exp_id = exp_dir.name
                    for log_file in train_logs_dir.glob(f"*{map_name}*/train.log"):
                        t_env_log = get_t_env_from_log(log_file)
                        if t_env_log:
                            t_env = t_env_log
                            break
            
            completion = t_env / t_max if t_max > 0 else 0
            
            win_rate = get_win_rate(info)
            reward = get_reward(info)
            
            # 选择完成度最高的实验
            if best_exp is None or completion > best_completion or (completion == best_completion and win_rate > best_win_rate):
                best_exp = {
                    'exp_id': exp_dir.name,
                    'config_dir': config_dir.name,
                    't_env': t_env,
                    't_max': t_max,
                    'completion': completion,
                    'win_rate': win_rate,
                    'reward': reward,
                    'info_path': str(info_path)
                }
                best_completion = completion
                best_win_rate = win_rate
    
    return best_exp

def get_status(exp_info, alg_path, map_name):
    """根据实验信息判断状态"""
    if exp_info is None:
        # 检查是否有模型文件
        if check_model_files(alg_path, map_name):
            return '已完成（仅模型）'
        return '未开始'
    
    completion = exp_info['completion']
    
    # 检查是否有模型文件
    has_model = check_model_files(alg_path, map_name)
    
    if completion >= COMPLETION_THRESHOLD:
        return '已完成'
    elif completion >= 0.50:  # 完成度>=50%，认为已完成
        return '已完成'
    elif has_model:  # 有模型文件但完成度<50%，也算已完成
        return '已完成（仅模型）'
    elif completion > 0:
        return '进行中'
    else:
        return '未开始'

def main():
    """主函数"""
    results = {}
    
    print("=" * 100)
    print("按算法整理地图完成情况（检查所有配置目录）")
    print("=" * 100)
    print()
    
    # 遍历每个算法
    for alg_name, alg_info in ALGORITHMS.items():
        alg_path = BASE_PATH / alg_name
        
        if not alg_path.exists():
            print(f"⚠️  算法目录不存在: {alg_name}")
            continue
        
        print(f"正在检查: {alg_name}...")
        results[alg_name] = {}
        
        # 遍历每个地图
        for map_name in ALL_MAPS:
            exp_info = find_best_experiment(alg_path, map_name)
            status = get_status(exp_info, alg_path, map_name)
            
            results[alg_name][map_name] = {
                'status': status,
                'exp_info': exp_info
            }
    
    print()
    print("=" * 100)
    print("统计结果（按算法）")
    print("=" * 100)
    print()
    
    # 按算法分类展示
    for alg_name in ALGORITHMS.keys():
        if alg_name not in results:
            continue
        
        alg_results = results[alg_name]
        
        # 统计完成情况
        completed = sum(1 for m in alg_results.values() if '已完成' in m['status'])
        in_progress = sum(1 for m in alg_results.values() if m['status'] == '进行中')
        not_started = sum(1 for m in alg_results.values() if m['status'] == '未开始')
        total = len(alg_results)
        
        print(f"\n{'=' * 100}")
        print(f"算法: {alg_name}")
        print(f"{'=' * 100}")
        print(f"总体进度: {completed}/{total} 已完成, {in_progress}/{total} 进行中, {not_started}/{total} 未开始")
        print()
        
        # 按状态分组显示
        status_groups = {
            '已完成': [],
            '已完成（仅模型）': [],
            '进行中': [],
            '未开始': []
        }
        
        for map_name, map_data in sorted(alg_results.items()):
            status = map_data['status']
            exp_info = map_data['exp_info']
            
            if '已完成' in status:
                if '仅模型' in status:
                    status_groups['已完成（仅模型）'].append((map_name, exp_info))
                else:
                    status_groups['已完成'].append((map_name, exp_info))
            elif status == '进行中':
                status_groups['进行中'].append((map_name, exp_info))
            else:
                status_groups['未开始'].append((map_name, None))
        
        # 显示已完成的地图
        if status_groups['已完成']:
            print("✅ 已完成:")
            for map_name, exp_info in sorted(status_groups['已完成'], key=lambda x: x[1]['win_rate'] if x[1] else 0, reverse=True):
                win_rate = exp_info['win_rate'] if exp_info else 0
                reward = exp_info['reward'] if exp_info else 0
                completion = exp_info['completion'] * 100 if exp_info else 0
                t_env = exp_info['t_env'] if exp_info else 0
                config_dir = exp_info.get('config_dir', 'N/A') if exp_info else 'N/A'
                print(f"  {map_name:8s} | 完成度: {completion:6.1f}% | 胜率: {win_rate:6.2%} | 奖励: {reward:8.2f} | t_env: {t_env:>10,} | 配置: {config_dir[:30]}")
        
        if status_groups['已完成（仅模型）']:
            print("\n✅ 已完成（仅模型文件）:")
            for map_name, exp_info in status_groups['已完成（仅模型）']:
                if exp_info:
                    completion = exp_info['completion'] * 100
                    print(f"  {map_name:8s} | 完成度: {completion:6.1f}% (有模型文件)")
                else:
                    print(f"  {map_name:8s} | (有模型文件)")
        
        # 显示进行中的地图
        if status_groups['进行中']:
            print("\n🔄 进行中:")
            for map_name, exp_info in sorted(status_groups['进行中'], key=lambda x: x[1]['completion'] if x[1] else 0, reverse=True):
                completion = exp_info['completion'] * 100 if exp_info else 0
                t_env = exp_info['t_env'] if exp_info else 0
                t_max = exp_info['t_max'] if exp_info else T_MAX_TARGET
                win_rate = exp_info['win_rate'] if exp_info else 0
                config_dir = exp_info.get('config_dir', 'N/A') if exp_info else 'N/A'
                print(f"  {map_name:8s} | 完成度: {completion:6.1f}% | 胜率: {win_rate:6.2%} | t_env: {t_env:>10,}/{t_max:>10,} | 配置: {config_dir[:30]}")
        
        # 显示未开始的地图
        if status_groups['未开始']:
            print("\n⏸️  未开始:")
            for map_name, _ in sorted(status_groups['未开始']):
                print(f"  {map_name}")
        
        print()
    
    # 汇总表格
    print("\n" + "=" * 100)
    print("汇总表格（按算法）")
    print("=" * 100)
    print()
    
    # 表头
    header = f"{'算法':<25} | " + " | ".join(f"{m:8s}" for m in ALL_MAPS)
    print(header)
    print("-" * len(header))
    
    # 每个算法的状态
    for alg_name in ALGORITHMS.keys():
        if alg_name not in results:
            continue
        
        alg_results = results[alg_name]
        status_line = f"{alg_name:<25} | "
        status_line += " | ".join(
            f"{alg_results.get(m, {}).get('status', '未开始'):8s}" 
            for m in ALL_MAPS
        )
        print(status_line)
    
    print()
    print("=" * 100)
    print("图例: 已完成 | 已完成（仅模型） | 进行中 | 未开始")
    print("=" * 100)

if __name__ == '__main__':
    main()

