#!/bin/zsh
# 环境配置脚本（Zsh版本）

echo "=========================================="
echo "HLSMAC 环境配置脚本 (Zsh)"
echo "=========================================="

# 1. 检查并设置SC2PATH
DEFAULT_SC2PATH="/share/project/ytz/StarCraftII"

if [ -z "$SC2PATH" ]; then
    if [ -d "$DEFAULT_SC2PATH" ]; then
        export SC2PATH="$DEFAULT_SC2PATH"
        echo "使用默认路径: SC2PATH=$SC2PATH"
        # 添加到.zshrc（如果还没有）
        if ! grep -q "SC2PATH.*StarCraftII" ~/.zshrc 2>/dev/null; then
            echo "" >> ~/.zshrc
            echo "# StarCraft II Path" >> ~/.zshrc
            echo "export SC2PATH=\"$DEFAULT_SC2PATH\"" >> ~/.zshrc
            echo "已添加到 ~/.zshrc"
        fi
    else
        echo "警告: 默认路径不存在: $DEFAULT_SC2PATH"
        exit 1
    fi
else
    echo "SC2PATH已设置: $SC2PATH"
fi

# 2. 检查地图文件
MAP_DIR="$SC2PATH/Maps"
if [ ! -d "$MAP_DIR" ]; then
    echo "创建Maps目录: $MAP_DIR"
    mkdir -p "$MAP_DIR"
fi

# 复制HLSMAC地图
PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
HLSMAC_MAPS="$PROJECT_DIR/Tactics_Maps/HLSMAC_Maps"

if [ -d "$HLSMAC_MAPS" ]; then
    echo "复制HLSMAC地图到 $MAP_DIR"
    cp -n "$HLSMAC_MAPS"/*.SC2Map "$MAP_DIR/" 2>/dev/null
    MAP_COUNT=$(ls -1 "$MAP_DIR"/*.SC2Map 2>/dev/null | wc -l)
    echo "地图文件已复制 (共 $MAP_COUNT 张地图)"
else
    echo "警告: 未找到HLSMAC地图目录: $HLSMAC_MAPS"
fi

# 3. 设置PYTHONPATH
PROJECT_ROOT="$(cd "$(dirname "$0")" && pwd)"
export PYTHONPATH="$PROJECT_ROOT:$PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT/RLalgs/dTAPE/src:$PYTHONPATH"
export PYTHONPATH="$PROJECT_ROOT/smac:$PYTHONPATH"

echo "PYTHONPATH已设置:"
echo "  $PYTHONPATH"

# 添加到.zshrc
if ! grep -q "PYTHONPATH.*StarCraft2_HLSMAC" ~/.zshrc 2>/dev/null; then
    echo "" >> ~/.zshrc
    echo "# HLSMAC Project" >> ~/.zshrc
    echo "export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"" >> ~/.zshrc
    echo "export PYTHONPATH=\"$PROJECT_ROOT/RLalgs/dTAPE/src:\$PYTHONPATH\"" >> ~/.zshrc
    echo "export PYTHONPATH=\"$PROJECT_ROOT/smac:\$PYTHONPATH\"" >> ~/.zshrc
    echo "已添加到 ~/.zshrc"
fi

# 4. 检查Python依赖
echo ""
echo "检查Python依赖..."
python3 -c "import torch; print('✓ PyTorch:', torch.__version__)" 2>/dev/null || echo "✗ 警告: PyTorch未安装"
python3 -c "import sacred; print('✓ Sacred: OK')" 2>/dev/null || echo "✗ 警告: Sacred未安装"
python3 -c "import yaml; print('✓ PyYAML: OK')" 2>/dev/null || echo "✗ 警告: PyYAML未安装"
python3 -c "import pysc2; print('✓ PySC2: OK')" 2>/dev/null || echo "✗ 警告: PySC2未安装"

echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "当前环境变量："
echo "  SC2PATH=$SC2PATH"
echo "  PYTHONPATH=$PYTHONPATH"
echo ""
echo "请运行以下命令使环境变量生效："
echo "  source ~/.zshrc"
echo ""
echo "或直接运行："
echo "  source setup_env_zsh.sh"
echo ""

