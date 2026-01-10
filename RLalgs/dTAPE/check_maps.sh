#!/bin/bash
# 检查地图文件是否存在

echo "检查 StarCraft II 地图文件..."
echo ""

if [ -f "/share/project/ytz/StarCraftII/Maps/pzyy_te.SC2Map" ]; then
    echo "✅ pzyy_te.SC2Map 存在"
    ls -lh /share/project/ytz/StarCraftII/Maps/pzyy_te.SC2Map
else
    echo "❌ pzyy_te.SC2Map 缺失"
fi

echo ""

if [ -f "/share/project/ytz/StarCraftII/Maps/ldtj_te.SC2Map" ]; then
    echo "✅ ldtj_te.SC2Map 存在"
    ls -lh /share/project/ytz/StarCraftII/Maps/ldtj_te.SC2Map
else
    echo "❌ ldtj_te.SC2Map 缺失"
fi

echo ""
echo "现有的 Tactics 地图："
ls -lh /share/project/ytz/StarCraftII/Maps/*_te.SC2Map 2>/dev/null

