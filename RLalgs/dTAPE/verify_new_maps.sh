#!/bin/bash
# 验证新地图的完整配置

echo "======================================"
echo "验证新地图配置"
echo "======================================"
echo ""

# 1. 检查地图文件
echo "1. 检查地图文件位置："
if [ -f "/share/project/ytz/StarCraftII/Maps/Tactics_Maps/pzyy.SC2Map" ]; then
    echo "  ✅ pzyy.SC2Map 存在 ($(du -h /share/project/ytz/StarCraftII/Maps/Tactics_Maps/pzyy.SC2Map | cut -f1))"
else
    echo "  ❌ pzyy.SC2Map 缺失"
fi

if [ -f "/share/project/ytz/StarCraftII/Maps/Tactics_Maps/ldtj.SC2Map" ]; then
    echo "  ✅ ldtj.SC2Map 存在 ($(du -h /share/project/ytz/StarCraftII/Maps/Tactics_Maps/ldtj.SC2Map | cut -f1))"
else
    echo "  ❌ ldtj.SC2Map 缺失"
fi

echo ""

# 2. 检查环境代码
echo "2. 检查环境代码："
if [ -f "/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac/smac/env/sc2_tactics/star36env_pzyy.py" ]; then
    echo "  ✅ star36env_pzyy.py 存在"
else
    echo "  ❌ star36env_pzyy.py 缺失"
fi

if [ -f "/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac/smac/env/sc2_tactics/star36env_ldtj.py" ]; then
    echo "  ✅ star36env_ldtj.py 存在"
else
    echo "  ❌ star36env_ldtj.py 缺失"
fi

echo ""

# 3. 检查环境注册
echo "3. 检查环境注册："
if grep -q "SC2TacticsPZYYEnv" /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/src/envs/__init__.py; then
    echo "  ✅ PZYY 已在 __init__.py 中注册"
else
    echo "  ❌ PZYY 未注册"
fi

if grep -q "SC2TacticsLDTJEnv" /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/src/envs/__init__.py; then
    echo "  ✅ LDTJ 已在 __init__.py 中注册"
else
    echo "  ❌ LDTJ 未注册"
fi

echo ""

# 4. 检查地图别名
echo "4. 检查地图别名映射："
if grep -q '"pzyy"' /share/project/ytz/RLproject/StarCraft2_HLSMAC/smac/smac/env/sc2_tactics/maps/__init__.py; then
    echo "  ✅ pzyy 别名已添加"
else
    echo "  ❌ pzyy 别名缺失"
fi

if grep -q '"ldtj"' /share/project/ytz/RLproject/StarCraft2_HLSMAC/smac/smac/env/sc2_tactics/maps/__init__.py; then
    echo "  ✅ ldtj 别名已添加"
else
    echo "  ❌ ldtj 别名缺失"
fi

echo ""
echo "======================================"
echo "验证完成！"
echo "======================================"
echo ""
echo "如果所有项目都显示 ✅，可以运行训练："
echo "  cd /share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE"
echo "  ./train_pzyy_gpu6_foreground.sh"
echo ""

