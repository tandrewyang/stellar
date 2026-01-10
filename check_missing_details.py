#!/usr/bin/env python3
"""
逐个检查每个算法的每个地图
检查缺失胜率和未完成的地图
"""
import os
import json
import re
from pathlib import Path

ALGORITHMS = ['CurriculumLearning', 'RewardShaping', 'dTAPE', 
              'EnhancedStateRepresentation', 'HierarchicalArchitecture', 'TransformerMixer']
ALL_MAPS = ['adcc', 'dhls', 'fkwz', 'gmzz', 'jctq', 'jdsr', 'sdjx', 'swct', 'tlhz', 'wwjz', 'wzsy', 'yqgz']
MAP_NAMES = {
    'adcc': '暗度陈仓',
    'dhls': '调虎离山',
    'fkwz': '反客为主',
    'gmzz': '关门捉贼',
    'jctq': '金蝉脱壳',
    'jdsr': '借刀杀人',
    'sdjx': '声东击西',
    'swct': '上屋抽梯',
    'tlhz': '偷梁换柱',
    'wwjz': '围魏救赵',
    'wzsy': '无中生有',
    'yqgz': '欲擒故纵'
}
BASE_PATH = Path('/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs')
T_MAX_TARGET = 2005000

def load_json(file_path):
    """加载JSON文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        return None

def get_win_rate(info):
    """从info.json中提取胜率"""
    if not info:
        return None
    
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
    
    return None

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
    except Exception as e:
        return 0
    
    return 0

def check_single_map(alg_dir, map_name):
    """检查单个地图的详细信息"""
    results_dir = alg_dir / 'results'
    sacred_dir = results_dir / 'sacred' / map_name
    
    if not sacred_dir.exists():
        return {
            'status': 'no_experiment',
            't_env': 0,
            't_max': T_MAX_TARGET,
            'completion': 0,
            'win_rate': None,
            'exp_count': 0
        }
    
    best_t_env = 0
    best_t_max = T_MAX_TARGET
    best_info = None
    exp_count = 0
    all_exps = []
    
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
            
            exp_count += 1
            
            info_path = exp_dir / 'info.json'
            config_path = exp_dir / 'config.json'
            cout_path = exp_dir / 'cout.txt'
            
            config = load_json(config_path) if config_path.exists() else {}
            t_max = config.get('t_max', T_MAX_TARGET)
            
            if t_max < 100000:
                continue
            
            # 优先从cout.txt读取
            t_env = get_t_env_from_cout(cout_path)
            
            # 如果cout.txt没有，尝试从info.json读取
            if t_env == 0:
                info = load_json(info_path) if info_path.exists() else {}
                if info:
                    t_env = info.get('t_env', 0)
            
            if t_env > 0 and t_env <= t_max * 1.5:
                info = load_json(info_path) if info_path.exists() else {}
                all_exps.append({
                    'exp_id': exp_dir.name,
                    'config_dir': config_dir.name,
                    't_env': t_env,
                    't_max': t_max,
                    'win_rate': get_win_rate(info),
                    'info_exists': info_path.exists(),
                    'cout_exists': cout_path.exists()
                })
                
                if t_env > best_t_env:
                    best_t_env = t_env
                    best_t_max = t_max
                    best_info = info
    
    completion = (best_t_env / best_t_max * 100) if best_t_max > 0 and best_t_env > 0 else 0
    win_rate = get_win_rate(best_info) if best_info else None
    
    return {
        'status': 'completed' if completion >= 95 else 'incomplete' if completion > 0 else 'not_started',
        't_env': best_t_env,
        't_max': best_t_max,
        'completion': completion,
        'win_rate': win_rate,
        'exp_count': exp_count,
        'all_exps': all_exps
    }

print('=' * 120)
print('逐个检查每个算法的每个地图')
print('=' * 120)
print()

all_issues = []

for alg_name in ALGORITHMS:
    alg_dir = BASE_PATH / alg_name
    if not alg_dir.exists():
        print(f"❌ {alg_name}: 算法目录不存在")
        continue
    
    print(f"\n{'='*120}")
    print(f"算法: {alg_name}")
    print(f"{'='*120}")
    
    incomplete_maps = []
    missing_win_rate_maps = []
    
    for map_name in ALL_MAPS:
        map_cn = MAP_NAMES.get(map_name, map_name)
        result = check_single_map(alg_dir, map_name)
        
        status_icon = '✅' if result['status'] == 'completed' else '⏳' if result['status'] == 'incomplete' else '❌'
        
        # 检查完成度
        if result['completion'] < 95:
            incomplete_maps.append((map_name, map_cn, result))
            print(f"{status_icon} {map_name} ({map_cn}): {result['completion']:.2f}% ({result['t_env']:,}/{result['t_max']:,}) - ⚠️ 未完成")
            if result['all_exps']:
                print(f"   实验详情:")
                for exp in result['all_exps'][:3]:  # 只显示前3个
                    print(f"     - {exp['config_dir']}/{exp['exp_id']}: {exp['t_env']:,}/{exp['t_max']:,} ({exp['t_env']/exp['t_max']*100:.2f}%)")
        
        # 检查胜率
        elif result['win_rate'] is None:
            missing_win_rate_maps.append((map_name, map_cn, result))
            print(f"{status_icon} {map_name} ({map_cn}): {result['completion']:.2f}% - ⚠️ 缺失胜率")
            if result['all_exps']:
                best_exp = max(result['all_exps'], key=lambda x: x['t_env'])
                print(f"   最佳实验: {best_exp['config_dir']}/{best_exp['exp_id']}")
                print(f"   info.json存在: {best_exp['info_exists']}, cout.txt存在: {best_exp['cout_exists']}")
        
        else:
            print(f"{status_icon} {map_name} ({map_cn}): {result['completion']:.2f}% - 胜率: {result['win_rate']*100:.2f}%")
    
    # 汇总
    print(f"\n{alg_name} 汇总:")
    print(f"  ✅ 已完成: {12 - len(incomplete_maps) - len(missing_win_rate_maps)}/12")
    if incomplete_maps:
        print(f"  ⏳ 未完成 ({len(incomplete_maps)}):")
        for map_name, map_cn, result in incomplete_maps:
            print(f"    - {map_name} ({map_cn}): {result['completion']:.2f}% ({result['t_env']:,}/{result['t_max']:,})")
        all_issues.append((alg_name, 'incomplete', incomplete_maps))
    
    if missing_win_rate_maps:
        print(f"  ⚠️  缺失胜率 ({len(missing_win_rate_maps)}):")
        for map_name, map_cn, result in missing_win_rate_maps:
            print(f"    - {map_name} ({map_cn}): {result['completion']:.2f}%")
        all_issues.append((alg_name, 'missing_win_rate', missing_win_rate_maps))

print(f"\n{'='*120}")
print("总体汇总")
print(f"{'='*120}")

if all_issues:
    print("\n发现的问题:")
    for alg_name, issue_type, maps in all_issues:
        if issue_type == 'incomplete':
            print(f"\n{alg_name} - 未完成的地图 ({len(maps)}):")
            for map_name, map_cn, result in maps:
                print(f"  {map_name} ({map_cn}): {result['completion']:.2f}% ({result['t_env']:,}/{result['t_max']:,})")
        elif issue_type == 'missing_win_rate':
            print(f"\n{alg_name} - 缺失胜率的地图 ({len(maps)}):")
            for map_name, map_cn, result in maps:
                print(f"  {map_name} ({map_cn}): {result['completion']:.2f}%")
else:
    print("\n✅ 所有算法所有地图都已完成，且都有胜率数据！")

print(f"\n{'='*120}")


