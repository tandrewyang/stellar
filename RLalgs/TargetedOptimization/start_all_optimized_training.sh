#!/bin/bash
# 启动所有四个地图的优化训练
# DHLS, YQGZ, TLHZ, FKWZ

cd "$(dirname "$0")"

echo "=========================================="
echo "启动所有四个地图的优化训练"
echo "=========================================="
echo ""
echo "GPU分配建议："
echo "  DHLS (调虎离山): GPU 4"
echo "  YQGZ (欲擒故纵): GPU 5"
echo "  TLHZ (偷梁换柱): GPU 6"
echo "  FKWZ (反客为主): GPU 7"
echo ""
echo "=========================================="
echo ""

# 检查GPU是否可用
check_gpu() {
    local gpu_id=$1
    local usage=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i $gpu_id 2>/dev/null)
    if [ $? -eq 0 ]; then
        usage=$(echo $usage | tr -d ' %')
        if [ "$usage" -lt 10 ]; then
            return 0  # GPU空闲
        else
            return 1  # GPU使用中
        fi
    else
        return 1  # 无法检查
    fi
}

# 启动训练函数
start_training() {
    local map_name=$1
    local gpu_id=$2
    local seed=$3
    
    echo "启动 $map_name 训练 (GPU $gpu_id)..."
    
    if check_gpu $gpu_id; then
        echo "  GPU $gpu_id 可用，启动训练..."
        # 后台运行，保存日志
        nohup bash train_single_map.sh $map_name $gpu_id $seed > "logs/${map_name}_gpu${gpu_id}.log" 2>&1 &
        local pid=$!
        echo "  $map_name 训练已启动 (PID: $pid, GPU: $gpu_id)"
        echo "  日志文件: logs/${map_name}_gpu${gpu_id}.log"
        echo ""
        return 0
    else
        echo "  警告: GPU $gpu_id 可能正在使用中，请检查"
        return 1
    fi
}

# 创建日志目录
mkdir -p logs

# 启动四个地图的训练
echo "开始启动训练..."
echo ""

# DHLS - GPU 4
start_training dhls 4 42

# YQGZ - GPU 5
start_training yqgz 5 42

# TLHZ - GPU 6
start_training tlhz 6 42

# FKWZ - GPU 7
start_training fkwz 7 42

echo "=========================================="
echo "所有训练已启动！"
echo "=========================================="
echo ""
echo "查看训练状态："
echo "  ps aux | grep train_single_map"
echo ""
echo "查看GPU使用情况："
echo "  nvidia-smi"
echo ""
echo "查看日志："
echo "  tail -f logs/dhls_gpu4.log"
echo "  tail -f logs/yqgz_gpu5.log"
echo "  tail -f logs/tlhz_gpu6.log"
echo "  tail -f logs/fkwz_gpu7.log"
echo ""

