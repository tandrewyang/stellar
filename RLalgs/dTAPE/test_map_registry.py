#!/usr/bin/env python3
"""测试地图参数是否正确注册"""
import sys
sys.path.insert(0, '/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac')

from smac.env.sc2_tactics.maps import get_map_params

print("测试地图参数注册...")
print("=" * 50)

# 测试 PZYY
try:
    pzyy_params = get_map_params("pzyy")
    print(f"✅ PZYY 地图参数获取成功:")
    print(f"   - n_agents: {pzyy_params.get('n_agents')}")
    print(f"   - n_enemies: {pzyy_params.get('n_enemies')}")
    print(f"   - map_type: {pzyy_params.get('map_type')}")
except Exception as e:
    print(f"❌ PZYY 地图参数获取失败: {e}")

print()

# 测试 LDTJ
try:
    ldtj_params = get_map_params("ldtj")
    print(f"✅ LDTJ 地图参数获取成功:")
    print(f"   - n_agents: {ldtj_params.get('n_agents')}")
    print(f"   - n_enemies: {ldtj_params.get('n_enemies')}")
    print(f"   - map_type: {ldtj_params.get('map_type')}")
except Exception as e:
    print(f"❌ LDTJ 地图参数获取失败: {e}")

print()
print("=" * 50)
print("测试完成！")

