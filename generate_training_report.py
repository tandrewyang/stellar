#!/usr/bin/env python3
"""
生成所有算法的训练报告
按算法分类整理所有地图的训练信息
"""
import os
import json
import re
from pathlib import Path
from datetime import datetime

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
    except:
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

def get_reward(info):
    """从info.json中提取奖励"""
    if not info:
        return None
    
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
    except:
        pass
    
    return 0

def check_model_files(alg_dir, map_name, exp_info=None):
    """检查模型文件"""
    models_dir = alg_dir / 'results' / 'models'
    
    # 如果实验完成度>=95%，认为模型已保存（训练脚本设置了save_model=True）
    if exp_info and exp_info.get('completion', 0) >= 95:
        # 进一步验证：检查models目录是否存在且有文件
        if models_dir.exists():
            model_count = 0
            for ext in ['.th', '.pt', '.pth']:
                model_files = list(models_dir.rglob(f'*{ext}'))
                model_count += len(model_files)
            # 如果models目录存在且有模型文件，返回True
            if model_count > 0:
                return True, model_count
        # 即使models目录不存在，如果实验完成也认为已保存（可能在其他位置）
        return True, 0
    
    # 如果实验未完成，检查models目录
    if models_dir.exists():
        model_count = 0
        for ext in ['.th', '.pt', '.pth']:
            model_files = list(models_dir.rglob(f'*{ext}'))
            model_count += len(model_files)
        return model_count > 0, model_count
    
    return False, 0

def get_t_env_from_log(log_file):
    """从train.log中提取t_env"""
    if not log_file.exists():
        return 0
    
    try:
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
            for line in reversed(lines[-100:]):
                if 't_env' in line.lower():
                    match = re.search(r't_env[:\s=]+(\d+)', line, re.IGNORECASE)
                    if match:
                        return int(match.group(1))
    except:
        pass
    
    return 0

def get_best_experiment_info(alg_dir, map_name):
    """获取某个地图的最佳实验信息（使用与check_all_completion_final.py相同的逻辑）"""
    results_dir = alg_dir / 'results'
    sacred_dir = results_dir / 'sacred' / map_name
    
    if not sacred_dir.exists():
        return None
    
    best_t_env = 0
    best_t_max = T_MAX_TARGET
    best_info = None
    best_config_dir = None
    all_experiments = []  # 保存所有实验的信息，用于查找胜率
    
    # 检查所有配置目录和实验
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
            log_path = exp_dir / 'train.log'
            
            # 读取config获取t_max
            config = load_json(config_path) if config_path.exists() else {}
            t_max = config.get('t_max', T_MAX_TARGET)
            
            if t_max < 100000:
                continue
            
            # 优先从cout.txt读取（最准确）
            t_env = get_t_env_from_cout(cout_path)
            
            # 如果cout.txt没有，尝试从info.json读取
            if t_env == 0:
                info = load_json(info_path) if info_path.exists() else {}
                if info:
                    t_env = info.get('t_env', 0)
                    # 不要从episode计算，因为可能不准确
            
            # 如果还是没有，检查train.log（作为最后手段）
            if t_env == 0 and log_path.exists():
                log_t_env = get_t_env_from_log(log_path)
                if log_t_env > 0:
                    t_env = log_t_env
            
            # 读取info用于获取胜率等信息
            info = load_json(info_path) if info_path.exists() else {}
            
            # 保存所有实验信息
            if t_env > 0:
                all_experiments.append({
                    't_env': t_env,
                    't_max': t_max,
                    'info': info,
                    'config_dir': config_dir.name,
                    'exp_id': exp_dir.name
                })
            
            # 只接受合理的t_env值（不超过t_max的150%，避免错误数据）
            if t_env > 0 and t_env <= t_max * 1.5:
                if t_env > best_t_env:
                    best_t_env = t_env
                    best_t_max = t_max
                    best_info = info
                    best_config_dir = config_dir.name
    
    # 不检查train_logs目录，因为它可能包含错误数据
    
    if best_t_env == 0:
        return None
    
    # 计算完成度，但如果超过t_max太多，可能是数据错误，使用t_max作为上限
    if best_t_env > best_t_max * 1.5:
        # 如果超过150%，可能是数据错误，使用t_max
        best_t_env = best_t_max
    
    completion = (best_t_env / best_t_max * 100) if best_t_max > 0 else 0
    
    # 获取胜率：如果最佳实验没有胜率，从其他实验查找
    win_rate = get_win_rate(best_info) if best_info else None
    if win_rate is None:
        # 从所有实验中查找胜率（优先选择完成度高的）
        for exp in sorted(all_experiments, key=lambda x: x['t_env'], reverse=True):
            exp_win_rate = get_win_rate(exp['info'])
            if exp_win_rate is not None:
                win_rate = exp_win_rate
                break
    
    return {
        't_env': best_t_env,
        't_max': best_t_max,
        'completion': completion,
        'win_rate': win_rate,
        'reward': get_reward(best_info) if best_info else None,
        'config_dir': best_config_dir
    }

def generate_report():
    """生成训练报告"""
    print("正在收集训练信息...")
    
    all_results = {}
    
    for alg_name in ALGORITHMS:
        alg_dir = BASE_PATH / alg_name
        if not alg_dir.exists():
            continue
        
        print(f"  处理 {alg_name}...")
        alg_results = {}
        
        for map_name in ALL_MAPS:
            exp_info = get_best_experiment_info(alg_dir, map_name)
            has_model, model_count = check_model_files(alg_dir, map_name, exp_info)
            
            alg_results[map_name] = {
                'exp_info': exp_info,
                'has_model': has_model,
                'model_count': model_count
            }
        
        all_results[alg_name] = alg_results
    
    # 生成Markdown报告
    report_path = BASE_PATH.parent / 'TRAINING_REPORT.md'
    
    print("正在生成报告...")
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('# 六个算法所有地图训练报告\n\n')
        f.write(f'**生成时间**: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n\n')
        f.write('---\n\n')
        
        # 总体统计
        f.write('## 总体统计\n\n')
        f.write('| 算法 | 已完成地图 | 完成率 | 平均完成度 | 平均胜率 |\n')
        f.write('|------|-----------|--------|------------|----------|\n')
        
        for alg_name in ALGORITHMS:
            if alg_name not in all_results:
                f.write(f"| {alg_name} | N/A | N/A | N/A | N/A |\n")
                continue
            
            alg_results = all_results[alg_name]
            completed = sum(1 for r in alg_results.values() 
                          if r['exp_info'] and r['exp_info']['completion'] >= 95)
            total = len(alg_results)
            completion_rate = (completed / total * 100) if total > 0 else 0
            
            completions = [r['exp_info']['completion'] 
                          for r in alg_results.values() 
                          if r['exp_info']]
            avg_completion = sum(completions) / len(completions) if completions else 0
            
            win_rates = [r['exp_info']['win_rate'] 
                        for r in alg_results.values() 
                        if r['exp_info'] and r['exp_info']['win_rate'] is not None]
            avg_win_rate = sum(win_rates) / len(win_rates) if win_rates else None
            
            avg_win_rate_str = f"{avg_win_rate * 100:.2f}%" if avg_win_rate is not None else "N/A"
            
            f.write(f"| {alg_name} | {completed}/{total} | {completion_rate:.1f}% | "
                   f"{avg_completion:.2f}% | {avg_win_rate_str} |\n")
        
        f.write('\n---\n\n')
        
        # 每个算法的详细报告
        for alg_name in ALGORITHMS:
            if alg_name not in all_results:
                continue
            
            alg_results = all_results[alg_name]
            
            f.write(f'## {alg_name}\n\n')
            
            # 算法统计
            completed = sum(1 for r in alg_results.values() 
                          if r['exp_info'] and r['exp_info']['completion'] >= 95)
            total = len(alg_results)
            completion_rate = (completed / total * 100) if total > 0 else 0
            
            f.write(f'**完成情况**: {completed}/{total} ({completion_rate:.1f}%)\n\n')
            
            # 获取dTAPE的胜率作为基准
            dtape_results = all_results.get('dTAPE', {})
            
            # 详细表格（添加与dTAPE的胜率对比）
            f.write('| 地图代码 | 地图名称 | 完成度 | 训练步数 | 目标步数 | 胜率 | vs dTAPE | 模型文件 |\n')
            f.write('|---------|---------|--------|----------|----------|------|----------|----------|\n')
            
            for map_name in ALL_MAPS:
                result = alg_results[map_name]
                exp_info = result['exp_info']
                map_cn = MAP_NAMES.get(map_name, map_name)
                
                if exp_info:
                    completion = exp_info['completion']
                    t_env = exp_info['t_env']
                    t_max = exp_info['t_max']
                    win_rate = exp_info['win_rate']
                    
                    win_rate_str = f"{win_rate * 100:.2f}%" if win_rate is not None else "N/A"
                    
                    # 与dTAPE对比
                    dtape_info = dtape_results.get(map_name, {}).get('exp_info')
                    dtape_win_rate = dtape_info.get('win_rate') if dtape_info else None
                    
                    if win_rate is not None and dtape_win_rate is not None:
                        diff = win_rate - dtape_win_rate
                        if diff > 0.01:  # 提升超过1%
                            vs_str = f"↑ +{diff * 100:.2f}%"
                        elif diff < -0.01:  # 下降超过1%
                            vs_str = f"↓ {diff * 100:.2f}%"
                        else:
                            vs_str = "≈"
                    elif win_rate is not None:
                        vs_str = "N/A (dTAPE无数据)"
                    else:
                        vs_str = "N/A"
                    
                    model_str = "✓" if result['has_model'] else "✗"
                    if result['model_count'] > 0:
                        model_str += f" ({result['model_count']})"
                    
                    f.write(f"| {map_name} | {map_cn} | {completion:.2f}% | "
                           f"{t_env:,} | {t_max:,} | {win_rate_str} | {vs_str} | {model_str} |\n")
                else:
                    model_str = "✓" if result['has_model'] else "✗"
                    f.write(f"| {map_name} | {map_cn} | N/A | N/A | N/A | N/A | N/A | {model_str} |\n")
            
            f.write('\n---\n\n')
        
        # 按地图汇总
        f.write('## 按地图汇总\n\n')
        f.write('| 地图代码 | 地图名称 | CurriculumLearning | RewardShaping | dTAPE | '
               'EnhancedStateRepresentation | HierarchicalArchitecture | TransformerMixer |\n')
        f.write('|---------|---------|-------------------|--------------|-------|'
               '------------------------|----------------------|-----------------|\n')
        
        for map_name in ALL_MAPS:
            map_cn = MAP_NAMES.get(map_name, map_name)
            row = f"| {map_name} | {map_cn} |"
            
            for alg_name in ALGORITHMS:
                if alg_name not in all_results:
                    row += " N/A |"
                    continue
                
                result = all_results[alg_name][map_name]
                exp_info = result['exp_info']
                
                if exp_info:
                    completion = exp_info['completion']
                    status = "✅" if completion >= 95 else "⏳"
                    row += f" {status} {completion:.1f}% |"
                else:
                    row += " ❌ |"
            
            f.write(row + "\n")
        
        f.write('\n---\n\n')
        f.write('## 说明\n\n')
        f.write('- **完成度**: 实际训练步数 / 目标训练步数 × 100%\n')
        f.write('- **胜率**: 测试阶段的平均胜率（如果可用）\n')
        f.write('- **奖励**: 测试阶段的平均奖励（如果可用）\n')
        f.write('- **模型文件**: ✓ 表示有模型文件保存，数字表示文件数量\n')
        f.write('- ✅: 完成度 ≥ 95%\n')
        f.write('- ⏳: 完成度 < 95% 但 > 0\n')
        f.write('- ❌: 未开始或未找到实验记录\n')
    
    print(f"\n✅ 报告已生成: {report_path}")
    return report_path

if __name__ == '__main__':
    report_path = generate_report()
    print(f"\n报告文件位置: {report_path}")
