#!/bin/bash
# 修复 Sacred Git 信息收集错误

echo "=========================================="
echo "修复 Sacred Git 信息收集问题"
echo "=========================================="

# 方案1: 设置环境变量禁用 Git 信息收集
export SACRED_DISABLE_GIT_INFO=1

# 方案2: 在代码中已设置 SETTINGS['SAVE_GIT_INFO'] = False

echo "✓ 已禁用 Sacred 的 Git 信息收集"
echo ""
echo "如果仍有问题，可以设置环境变量："
echo "  export SACRED_DISABLE_GIT_INFO=1"
echo ""
echo "或添加到 ~/.zshrc:"
echo "  echo 'export SACRED_DISABLE_GIT_INFO=1' >> ~/.zshrc"

