#!/bin/bash
# 监控jdsr地图训练日志的脚本

LOG_FILE="RLalgs/dTAPE/results/sacred/jdsr/ow_qmix_env=4_adam_td_lambda/1/cout.txt"

if [ ! -f "$LOG_FILE" ]; then
    echo "错误: 日志文件不存在: $LOG_FILE"
    echo "请确认jdsr训练是否正在运行"
    exit 1
fi

echo "=========================================="
echo "jdsr 地图训练监控"
echo "=========================================="
echo "日志文件: $LOG_FILE"
echo "按 Ctrl+C 退出监控"
echo "=========================================="
echo ""

# 显示最新训练指标
tail -20 "$LOG_FILE" | grep -E "target_mean|td_error_abs|test_battle_won_mean|test_reward_mean|test_dead_allies_mean|t_env" | tail -5

echo ""
echo "实时监控（最后20行）:"
echo "----------------------------------------"

# 实时监控日志
tail -f "$LOG_FILE" 2>/dev/null | while IFS= read -r line; do
    # 高亮显示关键信息
    if echo "$line" | grep -qE "test_battle_won_mean|td_error_abs|target_mean|Error|Failed|Exception"; then
        echo "$line" | sed 's/test_battle_won_mean/\x1b[32mtest_battle_won_mean\x1b[0m/g'
    else
        echo "$line"
    fi
done



