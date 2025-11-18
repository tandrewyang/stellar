#!/bin/bash
# 修复 protobuf 版本兼容性问题

echo "=========================================="
echo "修复 Protobuf 兼容性问题"
echo "=========================================="

# 检查当前版本
CURRENT_VERSION=$(python3 -c "import google.protobuf; print(google.protobuf.__version__)" 2>/dev/null || echo "unknown")
echo "当前 protobuf 版本: $CURRENT_VERSION"

# 方案1: 降级 protobuf 到 3.20.x（推荐）
echo ""
echo "方案1: 降级 protobuf 到 3.20.3（推荐）"
read -p "是否执行降级? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "正在降级 protobuf..."
    pip install protobuf==3.20.3 --force-reinstall
    echo "✓ protobuf 已降级到 3.20.3"
fi

# 方案2: 设置环境变量（临时方案）
echo ""
echo "方案2: 设置环境变量（临时方案，性能较慢）"
read -p "是否设置环境变量? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
    echo "export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python" >> ~/.zshrc
    echo "✓ 环境变量已设置（已添加到 ~/.zshrc）"
    echo "  注意: 这会使 protobuf 使用纯 Python 实现，速度较慢"
fi

echo ""
echo "=========================================="
echo "修复完成！"
echo "=========================================="
echo ""
echo "验证修复:"
python3 -c "from s2clientprotocol import sc2api_pb2; print('✓ s2clientprotocol 导入成功')" 2>&1
echo ""
echo "如果仍有问题，请尝试："
echo "  1. 重启终端"
echo "  2. source ~/.zshrc"
echo "  3. 重新运行训练脚本"

