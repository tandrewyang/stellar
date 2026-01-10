#!/bin/bash
# HLSMAC 12个地图的训练脚本
# RewardShaping算法

# HLSMAC 12个地图列表
MAPS=("adcc" "dhls" "fkwz" "gmzz" "jctq" "jdsr" "sdjx" "swct" "tlhz" "wwjz" "wzsy" "yqgz")

# 地图中文名称
declare -A MAP_NAMES
MAP_NAMES["adcc"]="暗度陈仓"
MAP_NAMES["dhls"]="调虎离山"
MAP_NAMES["fkwz"]="反客为主"
MAP_NAMES["gmzz"]="关门捉贼"
MAP_NAMES["jctq"]="金蝉脱壳"
MAP_NAMES["jdsr"]="借刀杀人"
MAP_NAMES["sdjx"]="声东击西"
MAP_NAMES["swct"]="上屋抽梯"
MAP_NAMES["tlhz"]="偷梁换柱"
MAP_NAMES["wwjz"]="围魏救赵"
MAP_NAMES["wzsy"]="无中生有"
MAP_NAMES["yqgz"]="欲擒故纵"

# 设置参数
GPU_ID=${1:-5}
SEED=${2:-42}
ALG_CONFIG="reward_shaping_qmix"
T_MAX=${T_MAX:-2005000}
BATCH_SIZE_RUN=${BATCH_SIZE_RUN:-1}

# 进入算法目录
cd "$(dirname "$0")"

# 设置protobuf环境变量（解决版本兼容性问题）
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python

# 设置SC2PATH（如果未设置）
if [ -z "$SC2PATH" ]; then
    DEFAULT_SC2PATH="/share/project/ytz/StarCraftII"
    if [ -d "$DEFAULT_SC2PATH" ]; then
        export SC2PATH="$DEFAULT_SC2PATH"
        echo "已自动设置 SC2PATH=$SC2PATH"
    else
        echo "错误: SC2PATH环境变量未设置，且默认路径不存在: $DEFAULT_SC2PATH"
        echo "请设置: export SC2PATH=/path/to/StarCraftII"
        exit 1
    fi
else
    echo "使用 SC2PATH=$SC2PATH"
fi

echo "=========================================="
echo "RewardShaping - HLSMAC 训练脚本"
echo "=========================================="
echo "算法: $ALG_CONFIG"
echo "GPU: $GPU_ID"
echo "种子: $SEED"
echo "训练步数: $T_MAX"
echo "SC2PATH: $SC2PATH"
echo "=========================================="
echo ""

# 训练所有地图
for map in "${MAPS[@]}"; do
    echo ""
    echo "=========================================="
    echo "开始训练地图: $map (${MAP_NAMES[$map]})"
    echo "=========================================="
    
    # 创建日志目录
    LOG_DIR="../results/train_logs/${map}_reward_shaping_qmix"
    mkdir -p "$LOG_DIR"
    
    # 运行训练
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py \
        --config=$ALG_CONFIG \
        --env-config=sc2te \
        with env_args.map_name=$map \
        seed=$SEED \
        t_max=$T_MAX \
        batch_size_run=$BATCH_SIZE_RUN \
        use_tensorboard=True \
        save_model=True \
        save_model_interval=500000 \
        2>&1 | tee "$LOG_DIR/train.log" &
    
    # 等待一下再启动下一个（如果使用同一个GPU）
    sleep 5s
    
    echo "地图 $map 训练已启动，日志保存在: $LOG_DIR/train.log"
done

echo ""
echo "=========================================="
echo "所有地图训练任务已启动"
echo "使用以下命令查看训练状态:"
echo "  tail -f ../results/train_logs/*/train.log"
echo "=========================================="

