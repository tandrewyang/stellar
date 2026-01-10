#!/usr/bin/env python3
"""
全面检查所有六个算法的所有地图完成情况
包括：运行状态、训练进度、模型文件、完成度
"""
import os
import json
import re
import subprocess
from pathlib import Path
from datetime import datetime
from collections import defaultdict

ALGORITHMS = ['CurriculumLearning', 'RewardShaping', 'dTAPE', 
              'EnhancedStateRepresentation', 'HierarchicalArchitecture', 'TransformerMixer']
ALL_MAPS = ['adcc', 'dhls', 'fkwz', 'gmzz', 'jctq', 'jdsr', 'sdjx', 'swct', 'tlhz', 'wwjz', 'wzsy', 'yqgz']
BASE_PATH = Path('/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs')
T_MAX_TARGET = 2005000
COMPLETION_THRESHOLD = 0.95  # 95%视为完成

def load_json(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except:
        return None

def get_running_processes():
    """获取当前正在运行的训练进程"""
    result = subprocess.run(['ps', 'aux'], capture_output=True, text=True)
    running = {}
    
    for line in result.stdout.split('\n'):
        if 'python' in line and 'main.py' in line and 'config=' in line:
            parts = line.split()
            if len(parts) > 1:
                pid = parts[1]
                
                # 提取算法和地图
                alg = None
                map_name = None
                gpu_id = None
                
                for a in ALGORITHMS:
                    if a.lower().replace('_', '').replace('-', '') in line.lower():
                        alg = a
                        break
                
                for m in ALL_MAPS:
                    if f'map_name={m}' in line or f'map_name={m}' in line:
                        map_name = m
                        break
                
                # 尝试从环境变量获取GPU
                try:
                    env_file = Path(f'/proc/{pid}/environ')
                    if env_file.exists():
                        with open(env_file, 'rb') as f:
                            content = f.read()
                            env_str = content.decode('utf-8', errors='ignore')
                            match = re.search(r'CUDA_VISIBLE_DEVICES=(\d+)', env_str)
                            if match:
                                gpu_id = int(match.group(1))
                except:
                    pass
                
                if alg and map_name:
                    key = f"{alg}_{map_name}"
                    running[key] = {
                        'pid': pid,
                        'gpu': gpu_id,
                        'cpu': parts[2] if len(parts) > 2 else '0',
                        'mem': parts[3] if len(parts) > 3 else '0'
                    }
    
    return running

def check_model_files(alg_dir, map_name):
    """检查模型文件是否存在"""
    models_dir = alg_dir / 'results' / 'models'
    if not models_dir.exists():
        return False, []
    
    model_files = []
    # 查找包含地图名称的模型文件
    for item in models_dir.rglob('*.th'):
        if map_name in item.name.lower() or map_name in str(item.parent).lower():
            model_files.append(str(item))
    
    # 也检查.pt和.pth文件
    for ext in ['.pt', '.pth']:
        for item in models_dir.rglob(f'*{ext}'):
            if map_name in item.name.lower() or map_name in str(item.parent).lower():
                model_files.append(str(item))
    
    return len(model_files) > 0, model_files

def get_t_env_from_cout(cout_file):
    """从cout.txt中提取t_env"""
    if not cout_file.exists():
        return 0
    
    try:
        with open(cout_file, 'r', encoding='utf-8', errors='ignore') as f:
            content = f.read()
            matches = re.findall(r't_env[:\s]+(\d+)', content, re.IGNORECASE)
            if matches:
                return max([int(m) for m in matches])
    except:
        pass
    
    return 0

def check_sacred_experiments(alg_dir, map_name):
    """检查sacred目录中的所有实验"""
    results_dir = alg_dir / 'results'
    sacred_dir = results_dir / 'sacred' / map_name
    
    if not sacred_dir.exists():
        return None, None
    
    best_t_env = 0
    best_t_max = T_MAX_TARGET
    best_exp_path = None
    
    for config_dir in sacred_dir.iterdir():
        if not config_dir.is_dir() or config_dir.name == '_sources':
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
            cout_path = exp_dir / 'cout.txt'
            
            # 读取config获取t_max
            config = load_json(config_path) if config_path.exists() else {}
            t_max = config.get('t_max', T_MAX_TARGET)
            
            if t_max < 100000:
                continue
            
            # 优先从cout.txt读取t_env（最准确）
            t_env = get_t_env_from_cout(cout_path)
            
            # 如果cout.txt没有，尝试从info.json读取
            if t_env == 0:
                info = load_json(info_path) if info_path.exists() else {}
                if info:
                    t_env = info.get('t_env', 0)
                    if t_env == 0:
                        episodes = info.get('episode', [])
                        if episodes and len(episodes) > 0:
                            last_episode = episodes[-1] if isinstance(episodes, list) else episodes
                            t_env = last_episode * 200
            
            # 如果还是没有，尝试从train.log读取
            if t_env == 0:
                log_file = exp_dir / 'train.log'
                if log_file.exists():
                    try:
                        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                            lines = f.readlines()
                            for line in reversed(lines[-200:]):
                                if 't_env' in line.lower():
                                    match = re.search(r't_env[:\s=]+(\d+)', line, re.IGNORECASE)
                                    if match:
                                        t_env = int(match.group(1))
                                        break
                    except:
                        pass
            
            if t_env > best_t_env:
                best_t_env = t_env
                best_t_max = t_max
                best_exp_path = f"{config_dir.name}/{exp_dir.name}"
    
    return best_t_env, best_t_max, best_exp_path

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
                for line in reversed(lines[-500:]):
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

def check_algorithm_map(alg_name, map_name, running_processes):
    """检查单个算法-地图组合的完成情况"""
    alg_dir = BASE_PATH / alg_name
    
    if not alg_dir.exists():
        return {
            'status': '算法目录不存在',
            't_env': 0,
            't_max': T_MAX_TARGET,
            'completion': 0,
            'has_model': False,
            'is_running': False,
            'gpu': None,
            'pid': None
        }
    
    # 检查是否正在运行
    key = f"{alg_name}_{map_name}"
    is_running = key in running_processes
    running_info = running_processes.get(key, {})
    
    # 检查sacred实验
    sacred_result = check_sacred_experiments(alg_dir, map_name)
    sacred_t_env = sacred_result[0] if sacred_result[0] is not None else 0
    sacred_t_max = sacred_result[1] if sacred_result[1] is not None else T_MAX_TARGET
    exp_path = sacred_result[2] if len(sacred_result) > 2 else None
    
    # 检查train_logs
    logs_t_env = check_train_logs(alg_dir, map_name)
    
    # 取最大值
    max_t_env = max(sacred_t_env, logs_t_env)
    t_max = sacred_t_max if sacred_result[0] is not None else T_MAX_TARGET
    
    # 检查模型文件
    has_model, model_files = check_model_files(alg_dir, map_name)
    
    # 计算完成度
    completion = (max_t_env / t_max * 100) if t_max > 0 and max_t_env > 0 else 0
    
    # 判断状态
    if is_running:
        status = '运行中'
    elif completion >= COMPLETION_THRESHOLD * 100:
        status = '已完成'
    elif has_model and completion >= 50:
        status = '已完成（有模型）'
    elif max_t_env > 0:
        status = '进行中'
    else:
        status = '未开始'
    
    return {
        'status': status,
        't_env': max_t_env,
        't_max': t_max,
        'completion': completion,
        'has_model': has_model,
        'model_files': model_files,
        'is_running': is_running,
        'gpu': running_info.get('gpu'),
        'pid': running_info.get('pid'),
        'exp_path': exp_path,
        'sacred_t_env': sacred_t_env,
        'logs_t_env': logs_t_env
    }

def main():
    print('=' * 120)
    print('全面检查所有六个算法的所有地图完成情况')
    print(f'检查时间: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('=' * 120)
    print()
    
    # 获取运行中的进程
    running_processes = get_running_processes()
    
    # 检查所有算法和地图
    all_results = {}
    summary = defaultdict(lambda: {'completed': 0, 'running': 0, 'in_progress': 0, 'not_started': 0})
    
    for alg_name in ALGORITHMS:
        all_results[alg_name] = {}
        for map_name in ALL_MAPS:
            result = check_algorithm_map(alg_name, map_name, running_processes)
            all_results[alg_name][map_name] = result
            
            # 更新统计
            if result['is_running']:
                summary[alg_name]['running'] += 1
            elif result['status'] == '已完成' or result['status'] == '已完成（有模型）':
                summary[alg_name]['completed'] += 1
            elif result['t_env'] > 0:
                summary[alg_name]['in_progress'] += 1
            else:
                summary[alg_name]['not_started'] += 1
    
    # 打印详细结果
    for alg_name in ALGORITHMS:
        print(f'\n{"=" * 120}')
        print(f'算法: {alg_name}')
        print(f'{"=" * 120}')
        
        alg_results = all_results[alg_name]
        stats = summary[alg_name]
        
        print(f'统计: 已完成 {stats["completed"]}/12 | 运行中 {stats["running"]}/12 | 进行中 {stats["in_progress"]}/12 | 未开始 {stats["not_started"]}/12')
        print()
        
        # 按状态分组
        completed = []
        running = []
        in_progress = []
        not_started = []
        
        for map_name in ALL_MAPS:
            result = alg_results[map_name]
            if result['is_running']:
                running.append((map_name, result))
            elif result['status'] == '已完成' or result['status'] == '已完成（有模型）':
                completed.append((map_name, result))
            elif result['t_env'] > 0:
                in_progress.append((map_name, result))
            else:
                not_started.append((map_name, result))
        
        # 显示运行中的
        if running:
            print('🔄 运行中:')
            for map_name, result in sorted(running):
                gpu_str = f"GPU {result['gpu']}" if result['gpu'] is not None else "GPU ?"
                pid_str = f"PID {result['pid']}" if result['pid'] else ""
                print(f'  {map_name:<8} | {result["completion"]:>6.2f}% | {result["t_env"]:>12,}/{result["t_max"]:>12,} | {gpu_str} {pid_str}')
            print()
        
        # 显示已完成的
        if completed:
            print('✅ 已完成:')
            for map_name, result in sorted(completed):
                model_str = '✓模型' if result['has_model'] else '✗模型'
                print(f'  {map_name:<8} | {result["completion"]:>6.2f}% | {result["t_env"]:>12,}/{result["t_max"]:>12,} | {model_str}')
            print()
        
        # 显示进行中的
        if in_progress:
            print('⏳ 进行中:')
            for map_name, result in sorted(in_progress, key=lambda x: x[1]['completion'], reverse=True):
                model_str = '✓模型' if result['has_model'] else '✗模型'
                exp_str = f" | {result['exp_path']}" if result['exp_path'] else ""
                print(f'  {map_name:<8} | {result["completion"]:>6.2f}% | {result["t_env"]:>12,}/{result["t_max"]:>12,} | {model_str}{exp_str}')
            print()
        
        # 显示未开始的
        if not_started:
            print('⏸️  未开始:')
            for map_name, result in sorted(not_started):
                print(f'  {map_name}')
            print()
    
    # 汇总表格
    print('\n' + '=' * 120)
    print('汇总表格（所有算法 × 所有地图）')
    print('=' * 120)
    print()
    
    # 表头
    header = f"{'算法':<25} | " + " | ".join(f"{m:>6s}" for m in ALL_MAPS)
    print(header)
    print('-' * len(header))
    
    # 每个算法的状态
    for alg_name in ALGORITHMS:
        alg_results = all_results[alg_name]
        status_line = f"{alg_name:<25} | "
        status_chars = []
        for map_name in ALL_MAPS:
            result = alg_results[map_name]
            if result['is_running']:
                status_chars.append('🔄')
            elif result['status'] == '已完成' or result['status'] == '已完成（有模型）':
                status_chars.append('✅')
            elif result['t_env'] > 0:
                status_chars.append('⏳')
            else:
                status_chars.append('⏸️ ')
        status_line += " | ".join(f"{s:>6s}" for s in status_chars)
        print(status_line)
    
    print()
    print('=' * 120)
    print('图例: 🔄 运行中 | ✅ 已完成 | ⏳ 进行中 | ⏸️  未开始')
    print('=' * 120)
    
    # 总体统计
    total_completed = sum(s['completed'] for s in summary.values())
    total_running = sum(s['running'] for s in summary.values())
    total_in_progress = sum(s['in_progress'] for s in summary.values())
    total_not_started = sum(s['not_started'] for s in summary.values())
    total_tasks = len(ALGORITHMS) * len(ALL_MAPS)
    
    print()
    print('=' * 120)
    print('总体统计')
    print('=' * 120)
    print(f'总任务数: {total_tasks} (6算法 × 12地图)')
    print(f'已完成: {total_completed}/{total_tasks} ({total_completed/total_tasks*100:.1f}%)')
    print(f'运行中: {total_running}/{total_tasks}')
    print(f'进行中: {total_in_progress}/{total_tasks}')
    print(f'未开始: {total_not_started}/{total_tasks}')
    print('=' * 120)

if __name__ == '__main__':
    main()




