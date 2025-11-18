#!/bin/zsh
# 快速设置环境变量（Zsh版本）
# 使用方法: source set_env.sh

export SC2PATH="/share/project/ytz/StarCraftII"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC:$PYTHONPATH"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE/src:$PYTHONPATH"
export PYTHONPATH="/share/project/ytz/RLproject/StarCraft2_HLSMAC/smac:$PYTHONPATH"

echo "环境变量已设置:"
echo "  SC2PATH=$SC2PATH"
echo "  PYTHONPATH=$PYTHONPATH"
