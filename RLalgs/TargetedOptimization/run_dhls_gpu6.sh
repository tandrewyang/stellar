#!/bin/bash
# 启动dhls地图训练 (GPU6) - 前台运行，实时显示
# dhls特点: 16 vs 11, 200步, NydusCanal（虫洞）机制
# 优化: 机制感知

cd "$(dirname "$0")"
echo "启动 dhls 训练 (GPU6) - 前台运行"
echo "按 Ctrl+C 可停止训练"
echo ""
bash train_single_map.sh dhls 6 42

