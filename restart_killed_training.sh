#!/bin/bash
# 重新启动被终止的GPU 4-7训练任务
# 将从最近的checkpoint恢复训练

echo "════════════════════════════════════════════════════════════════"
echo "重启GPU 4-7上被终止的训练任务"
echo "════════════════════════════════════════════════════════════════"
echo ""

PAUSED_FILE="/share/project/ytz/RLproject/StarCraft2_HLSMAC/paused_training.txt"

if [ ! -f "$PAUSED_FILE" ]; then
    echo "❌ 错误: 找不到训练信息文件 $PAUSED_FILE"
    exit 1
fi

echo "从文件读取训练任务信息..."
echo ""

# 设置环境变量
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
export SC2PATH="/share/project/ytz/StarCraftII"

# 解析并重启任务
while IFS='|' read -r line; do
    if [[ "$line" =~ ^PID:([0-9]+)\|GPU:([0-9]+)\|CMD:(.+)$ ]]; then
        gpu="${BASH_REMATCH[2]}"
        full_cmd="${BASH_REMATCH[3]}"
        
        # 只处理GPU 4-7
        if [[ ! "$gpu" =~ ^[4567]$ ]]; then
            continue
        fi
        
        # 提取关键信息
        map=$(echo "$full_cmd" | grep -oP 'map_name=\K[^ ]+' | head -1)
        config=$(echo "$full_cmd" | grep -oP '\-\-config=\K[^ ]+' | head -1)
        
        echo "重启任务: GPU $gpu | $map | $config"
        
        # 提取工作目录（从命令路径推断）
        if [[ "$config" == "curriculum_qmix" ]]; then
            work_dir="/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/CurriculumLearning"
        elif [[ "$config" == "reward_shaping_qmix" ]]; then
            work_dir="/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/RewardShaping"
        elif [[ "$config" == "d_tape" ]]; then
            work_dir="/share/project/ytz/RLproject/StarCraft2_HLSMAC/RLalgs/dTAPE"
        else
            echo "  ⚠️ 未知配置: $config，跳过"
            continue
        fi
        
        # 创建日志目录
        log_dir="/share/project/ytz/RLproject/StarCraft2_HLSMAC/results/train_logs/${map}_${config}"
        mkdir -p "$log_dir"
        
        # 启动训练（后台运行）
        cd "$work_dir" || continue
        
        echo "  启动中..."
        CUDA_VISIBLE_DEVICES=$gpu nohup python3 src/main.py \
            --config=$config \
            --env-config=sc2te \
            with env_args.map_name=$map \
            seed=42 \
            t_max=2005000 \
            batch_size_run=1 \
            use_tensorboard=True \
            save_model=True \
            save_model_interval=500000 \
            > "$log_dir/train.log" 2>&1 &
        
        NEW_PID=$!
        echo "  ✅ 已启动 (PID: $NEW_PID)"
        
        # 等待一下再启动下一个
        sleep 5
    fi
done < "$PAUSED_FILE"

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "重启完成"
echo "════════════════════════════════════════════════════════════════"
echo ""
echo "查看GPU状态:"
nvidia-smi
echo ""
echo "注意: 训练将从最近保存的checkpoint继续"
echo "如需删除暂停信息文件："
echo "  rm $PAUSED_FILE"


