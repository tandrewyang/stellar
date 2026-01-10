#!/usr/bin/env python3
"""
综合检查所有算法的所有地图的完成情况
同时检查sacred目录和train_logs目录
"""
import os
import json
import re
from pathlib import Path

ALGORITHMS = ['CurriculumLearning', 'RewardShaping', 'dTAPE', 
              'EnhancedStateRepresentation', 'HierarchicalArchitecture', 'TransformerMixer']
ALL_MAPS = ['adcc', 'dhls', 'fkwz', 'gmzz', 'jctq', 'jdsr', 'sdjx', 'swct', 'tlhz', 'wwjz', 'wzsy', 'yqgz']
BASE_PATH = Path('/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs')
T_MAX_TARGET = 2005000

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def check_sacred_experiments(alg_dir, map_name):
    """检查sacred目录中的所有实验"""
    results_dir = alg_dir / 'results'
    sacred_dir = results_dir / 'sacred' / map_name
    
    if not sacred_dir.exists():
        return None
    
    best_t_env = 0
    best_t_max = T_MAX_TARGET
    
    for config_dir in sacred_dir.iterdir():
        if not config_dir.is_dir():
            continue
        
        for exp_dir in config_dir.iterdir():
            if not exp_dir.is_dir():
                continue
            
            try:
                int(exp_dir.name)
            except ValueError:
                continue
            
            info_path = exp_dir / 'info.json'
            config_path = exp_dir / 'config.json'
            
            # 即使info.json不存在或读取失败，也继续检查（可能cout.txt中有数据）
            info = load_json(info_path) if info_path.exists() else {}
            
            config = load_json(config_path) if config_path.exists() else {}
            t_max = config.get('t_max', T_MAX_TARGET)
            
            if t_max < 100000:
                continue
            
            t_env = info.get('t_env', 0) if info else 0
            if t_env == 0 and info:
                episodes = info.get('episode', [])
                if episodes and len(episodes) > 0:
                    last_episode = episodes[-1] if isinstance(episodes, list) else episodes
                    t_env = last_episode * 200
            
            # 检查日志文件
            log_file = exp_dir / 'train.log'
            if log_file.exists():
                try:
                    with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                        lines = f.readlines()
                        for line in reversed(lines[-100:]):
                            if 't_env' in line.lower():
                                match = re.search(r't_env[:\s=]+(\d+)', line, re.IGNORECASE)
                                if match:
                                    log_t_env = int(match.group(1))
                                    if log_t_env > t_env:
                                        t_env = log_t_env
                                    break
                except:
                    pass
            
            # 检查cout.txt文件（Sacred的输出文件）
            cout_file = exp_dir / 'cout.txt'
            if cout_file.exists():
                try:
                    with open(cout_file, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                        # 查找所有t_env值
                        matches = re.findall(r't_env[:\s]+(\d+)', content, re.IGNORECASE)
                        if matches:
                            cout_t_envs = [int(m) for m in matches]
                            if cout_t_envs:
                                cout_t_env = max(cout_t_envs)
                                if cout_t_env > t_env:
                                    t_env = cout_t_env
                except:
                    pass
            
            if t_env > best_t_env:
                best_t_env = t_env
                best_t_max = t_max
    
    return best_t_env, best_t_max

def check_train_logs(alg_dir, map_name):
    """检查train_logs目录"""
    train_logs_dir = alg_dir / 'results' / 'train_logs'
    
    if not train_logs_dir.exists():
        return 0
    
    max_t_env = 0
    
    log_dirs = list(train_logs_dir.glob(f'*{map_name}*'))
    for log_dir in log_dirs:
        log_file = log_dir / 'train.log'
        if not log_file.exists():
            continue
        
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                for line in reversed(lines[-200:]):
                    if 't_env' in line.lower():
                        match = re.search(r't_env[:\s=]+(\d+)', line, re.IGNORECASE)
                        if match:
                            t_env = int(match.group(1))
                            if t_env > max_t_env:
                                max_t_env = t_env
                            break
        except:
            pass
    
    return max_t_env

print('=' * 100)
print('综合检查所有算法的所有地图完成情况（sacred + train_logs）')
print('=' * 100)
print()

all_results = {}

for alg in ALGORITHMS:
    alg_dir = BASE_PATH / alg
    
    alg_results = {}
    
    for map_name in ALL_MAPS:
        # 检查sacred目录
        sacred_result = check_sacred_experiments(alg_dir, map_name)
        sacred_t_env = sacred_result[0] if sacred_result else 0
        sacred_t_max = sacred_result[1] if sacred_result else T_MAX_TARGET
        
        # 检查train_logs目录
        logs_t_env = check_train_logs(alg_dir, map_name)
        
        # 取最大值
        max_t_env = max(sacred_t_env, logs_t_env)
        t_max = sacred_t_max if sacred_result else T_MAX_TARGET
        
        completion = (max_t_env / t_max * 100) if t_max > 0 and max_t_env > 0 else 0
        
        alg_results[map_name] = {
            't_env': max_t_env,
            't_max': t_max,
            'completion': completion,
            'sacred_t_env': sacred_t_env,
            'logs_t_env': logs_t_env
        }
    
    all_results[alg] = alg_results

# 打印详细结果
for alg in ALGORITHMS:
    if alg not in all_results:
        continue
    
    print(f'\n{alg}:')
    print('-' * 100)
    
    alg_results = all_results[alg]
    completed = 0
    incomplete = []
    
    for map_name in ALL_MAPS:
        result = alg_results[map_name]
        completion = result['completion']
        t_env = result['t_env']
        t_max = result['t_max']
        
        if completion >= 95:
            completed += 1
            status = '✅'
        else:
            status = '❌'
            incomplete.append((map_name, completion, t_env, t_max))
        
        print(f'  {status} {map_name:<8} {completion:>6.2f}% ({t_env:>12,}/{t_max:>12,})')
    
    print(f'\n  总结: {completed}/12 已完成')
    if incomplete:
        print(f'  未完成 ({len(incomplete)}):')
        for map_name, comp, t_env, t_max in incomplete:
            print(f'    {map_name}: {comp:.2f}% ({t_env:,}/{t_max:,})')
    else:
        print(f'  ✅ 所有12个地图都已完成!')

