#!/bin/bash
# 恢复GPU 5,6,7上被暂停的训练任务
# 使用方法: ./resume_training.sh

echo "════════════════════════════════════════════════════════════════"
echo "恢复被暂停的训练任务"
echo "════════════════════════════════════════════════════════════════"
echo ""

PAUSED_FILE="/share/project/ytz/RLproject/StarCraft2_HLSMAC/paused_training.txt"

if [ ! -f "$PAUSED_FILE" ]; then
    echo "❌ 错误: 找不到暂停信息文件 $PAUSED_FILE"
    exit 1
fi

echo "读取暂停的进程信息..."
echo ""

PIDS_TO_RESUME=()
while IFS='|' read -r line; do
    if [[ "$line" =~ ^PID:([0-9]+) ]]; then
        pid="${BASH_REMATCH[1]}"
        PIDS_TO_RESUME+=($pid)
    fi
done < "$PAUSED_FILE"

if [ ${#PIDS_TO_RESUME[@]} -eq 0 ]; then
    echo "❌ 未找到需要恢复的进程"
    exit 1
fi

echo "找到 ${#PIDS_TO_RESUME[@]} 个被暂停的进程"
echo ""

for pid in "${PIDS_TO_RESUME[@]}"; do
    # 检查进程是否还存在
    if ps -p $pid > /dev/null 2>&1; then
        status=$(ps -o stat= -p $pid)
        if [[ "$status" == *"T"* ]]; then
            # 获取进程信息
            gpu=$(cat /proc/$pid/environ 2>/dev/null | tr '\0' '\n' | grep "CUDA_VISIBLE_DEVICES" | cut -d'=' -f2)
            cmdline=$(cat /proc/$pid/cmdline 2>/dev/null | tr '\0' ' ')
            map=$(echo "$cmdline" | grep -oP 'map_name=\K[^ ]+' | head -1)
            
            echo "恢复 PID $pid (GPU $gpu, Map: $map)..."
            kill -CONT $pid
            
            # 验证恢复
            sleep 1
            new_status=$(ps -o stat= -p $pid)
            if [[ "$new_status" != *"T"* ]]; then
                echo "  ✅ 成功恢复"
            else
                echo "  ⚠️  恢复可能失败，请手动检查"
            fi
        else
            echo "⚠️  PID $pid 未处于暂停状态 (状态: $status)"
        fi
    else
        echo "⚠️  PID $pid 进程已不存在"
    fi
done

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "恢复操作完成"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "查看GPU状态:"
nvidia-smi

echo ""
echo "如果一切正常，可以删除暂停信息文件："
echo "  rm $PAUSED_FILE"


