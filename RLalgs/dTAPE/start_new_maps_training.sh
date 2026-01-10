#!/bin/bash
# 启动 PZYY 和 LDTJ 两个新地图的 dTAPE 训练

echo "=========================================="
echo "启动新地图训练"
echo "=========================================="
echo "地图 1: PZYY (抛砖引玉) - GPU 6"
echo "地图 2: LDTJ (李代桃僵) - GPU 7"
echo "=========================================="

# 进入脚本目录
cd "$(dirname "$0")"

# 启动 PZYY 训练
echo ""
echo "启动 PZYY 训练..."
./train_pzyy_gpu6.sh
sleep 5

# 启动 LDTJ 训练
echo ""
echo "启动 LDTJ 训练..."
./train_ldtj_gpu7.sh
sleep 2

echo ""
echo "=========================================="
echo "所有训练任务已启动！"
echo "=========================================="
echo ""
echo "查看运行状态："
echo "  ps aux | grep 'main.py' | grep -E 'pzyy|ldtj'"
echo ""
echo "查看 PZYY 日志："
echo "  tail -f results/train_logs/pzyy_dtape/train_*.log"
echo ""
echo "查看 LDTJ 日志："
echo "  tail -f results/train_logs/ldtj_dtape/train_*.log"
echo ""
echo "停止所有训练："
echo "  pkill -f 'map_name=pzyy'"
echo "  pkill -f 'map_name=ldtj'"
echo ""

