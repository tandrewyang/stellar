#!/bin/bash
# 评测所有HLSMAC地图的脚本

# HLSMAC 12个地图列表
MAPS=("adcc" "dhls" "fkwz" "gmzz" "jctq" "jdsr" "sdjx" "swct" "tlhz" "wwjz" "wzsy" "yqgz")

# 参数
RESULTS_DIR=${RESULTS_DIR:-"../../results"}
ALG_CONFIG=${ALG_CONFIG:-"d_tape"}
GPU_ID=${GPU_ID:-0}
TEST_NEPISODE=${TEST_NEPISODE:-32}
LOAD_STEP=${LOAD_STEP:-0}  # 0表示加载最新的模型

# 进入算法目录
cd "$(dirname "$0")/RLalgs/dTAPE"

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
fi

echo "=========================================="
echo "评测所有HLSMAC地图"
echo "=========================================="
echo "结果目录: $RESULTS_DIR"
echo "算法: $ALG_CONFIG"
echo "测试回合数: $TEST_NEPISODE"
echo "=========================================="
echo ""

# 创建评测结果目录
EVAL_DIR="../../results/evaluation_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$EVAL_DIR"

# 评测每个地图
for map in "${MAPS[@]}"; do
    echo ""
    echo "=========================================="
    echo "评测地图: $map"
    echo "=========================================="
    
    # 查找模型文件
    MODEL_DIR="$RESULTS_DIR/${map}_${ALG_CONFIG}"
    if [ $LOAD_STEP -eq 0 ]; then
        # 查找最新的模型
        CHECKPOINT=$(find "$MODEL_DIR" -name "*.pt" -type f | sort -V | tail -1)
    else
        CHECKPOINT="$MODEL_DIR/models/episode_${LOAD_STEP}.pt"
    fi
    
    if [ -z "$CHECKPOINT" ] || [ ! -f "$CHECKPOINT" ]; then
        echo "警告: 未找到地图 $map 的模型文件，跳过"
        echo " 查找路径: $MODEL_DIR"
        continue
    fi
    
    echo "使用模型: $CHECKPOINT"
    
    # 运行评测
    CUDA_VISIBLE_DEVICES=$GPU_ID python3 src/main.py \
        --config=$ALG_CONFIG \
        --env-config=sc2te \
        with env_args.map_name=$map \
        evaluate=True \
        checkpoint_path="$CHECKPOINT" \
        load_step=$LOAD_STEP \
        test_nepisode=$TEST_NEPISODE \
        test_greedy=True \
        save_replay=True \
        env_args.replay_dir="$SC2PATH/Replays" \
        env_args.replay_prefix="${map}_eval" \
        2>&1 | tee "$EVAL_DIR/${map}_eval.log"
    
    echo "地图 $map 评测完成"
done

echo ""
echo "=========================================="
echo "所有地图评测完成！"
echo "结果保存在: $EVAL_DIR"
echo "回放文件保存在: $SC2PATH/Replays/"
echo "=========================================="

