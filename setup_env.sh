#!/bin/bash
# 环境配置脚本

echo "=========================================="
echo "HLSMAC 环境配置脚本"
echo "=========================================="

# 1. 检查并设置SC2PATH
DEFAULT_SC2PATH="/share/project/ytz/StarCraftII"

if [ -z "$SC2PATH" ]; then
    if [ -d "$DEFAULT_SC2PATH" ]; then
        export SC2PATH="$DEFAULT_SC2PATH"
        echo "使用默认路径: SC2PATH=$SC2PATH"
        # 添加到.bashrc（如果还没有）
        if ! grep -q "SC2PATH.*StarCraftII" ~/.bashrc 2>/dev/null; then
            echo "export SC2PATH=\"$DEFAULT_SC2PATH\"" >> ~/.bashrc
            echo "已添加到 ~/.bashrc"
        fi
    else
        echo "警告: 默认路径不存在: $DEFAULT_SC2PATH"
        echo "请设置SC2PATH环境变量"
        read -p "请输入StarCraftII安装路径: " sc2_path
        if [ -d "$sc2_path" ]; then
            export SC2PATH="$sc2_path"
            echo "export SC2PATH=\"$sc2_path\"" >> ~/.bashrc
            echo "已设置SC2PATH=$SC2PATH"
        else
            echo "错误: 路径不存在: $sc2_path"
            exit 1
        fi
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
    echo "地图文件已复制"
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

# 添加到.bashrc
if ! grep -q "PYTHONPATH.*StarCraft2_HLSMAC" ~/.bashrc 2>/dev/null; then
    echo "" >> ~/.bashrc
    echo "# HLSMAC Project" >> ~/.bashrc
    echo "export PYTHONPATH=\"$PROJECT_ROOT:\$PYTHONPATH\"" >> ~/.bashrc
    echo "export PYTHONPATH=\"$PROJECT_ROOT/RLalgs/dTAPE/src:\$PYTHONPATH\"" >> ~/.bashrc
    echo "export PYTHONPATH=\"$PROJECT_ROOT/smac:\$PYTHONPATH\"" >> ~/.bashrc
    echo "已添加到 ~/.bashrc"
fi

# 4. 检查Python依赖
echo ""
echo "检查Python依赖..."
python3 -c "import torch; print('PyTorch:', torch.__version__)" 2>/dev/null || echo "警告: PyTorch未安装"
python3 -c "import sacred; print('Sacred: OK')" 2>/dev/null || echo "警告: Sacred未安装"
python3 -c "import yaml; print('PyYAML: OK')" 2>/dev/null || echo "警告: PyYAML未安装"
python3 -c "import pysc2; print('PySC2: OK')" 2>/dev/null || echo "警告: PySC2未安装"

# 5. 检测当前shell并给出相应提示
CURRENT_SHELL=$(basename "$SHELL" 2>/dev/null || echo "unknown")
echo ""
echo "=========================================="
echo "环境配置完成！"
echo "=========================================="
echo ""
echo "当前Shell: $CURRENT_SHELL"
echo "当前环境变量："
echo "  SC2PATH=$SC2PATH"
echo "  PYTHONPATH=$PYTHONPATH"
echo ""

if [[ "$CURRENT_SHELL" == "zsh" ]]; then
    echo "检测到使用Zsh，请运行："
    echo "  source ~/.zshrc"
    echo ""
    echo "或使用Zsh专用脚本："
    echo "  bash setup_env_zsh.sh"
elif [[ "$CURRENT_SHELL" == "bash" ]]; then
    echo "请运行以下命令使环境变量生效："
    echo "  source ~/.bashrc"
else
    echo "请手动source相应的配置文件："
    echo "  source ~/.${CURRENT_SHELL}rc"
fi
echo ""

