#!/bin/bash
#
# MoE Hyperparameter Search Launcher
# ===================================
#
# 用法:
#   ./start_moe_search.sh [options]
#
# 选项:
#   --dry-run     只打印命令不执行
#   --resume FILE 从检查点恢复
#   --seeds N     每个配置的种子数 (默认10)
#   --gpu ID      使用的GPU (默认3)
#

set -e

# 默认参数
GPU_ID="3"
NUM_SEEDS=10
DRY_RUN=""
RESUME=""
MODE="grouped"

# 解析命令行参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --dry-run)
            DRY_RUN="--dry-run"
            shift
            ;;
        --resume)
            RESUME="--resume $2"
            shift 2
            ;;
        --seeds)
            NUM_SEEDS="$2"
            shift 2
            ;;
        --gpu)
            GPU_ID="$2"
            shift 2
            ;;
        --mode)
            MODE="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# 显示配置
echo "========================================"
echo "MoE Hyperparameter Search"
echo "========================================"
echo "GPU: $GPU_ID"
echo "Seeds per config: $NUM_SEEDS"
echo "Mode: $MODE"
echo "Dry run: ${DRY_RUN:-No}"
echo "Resume: ${RESUME:-No}"
echo "========================================"

# 计算预估实验数量
if [ "$MODE" == "grouped" ]; then
    # 组1: 5*4=20, 组2: 4*3*3=36, 组3: 4
    TOTAL_CONFIGS=60
else
    # 完整网格: 5*4*4*3*3*4=2880
    TOTAL_CONFIGS=2880
fi

TOTAL_EXPS=$((TOTAL_CONFIGS * NUM_SEEDS))
echo "Estimated experiments: $TOTAL_EXPS"
echo "========================================"

# 确认执行
if [ -z "$DRY_RUN" ]; then
    read -p "Start search? (y/N): " confirm
    if [ "$confirm" != "y" ] && [ "$confirm" != "Y" ]; then
        echo "Aborted."
        exit 0
    fi
fi

# 设置CUDA设备
export CUDA_VISIBLE_DEVICES=$GPU_ID

# 创建日志目录
LOG_DIR="moe_search_logs"
mkdir -p $LOG_DIR
LOG_FILE="$LOG_DIR/search_$(date +%Y%m%d_%H%M%S).log"

echo "Starting search..."
echo "Log file: $LOG_FILE"

# 运行搜索
python run_moe_hyperparam_search.py \
    --gpu "$GPU_ID" \
    --seeds "$NUM_SEEDS" \
    --mode "$MODE" \
    $DRY_RUN \
    $RESUME \
    2>&1 | tee "$LOG_FILE"

echo ""
echo "========================================"
echo "Search completed!"
echo "========================================"
echo "Log: $LOG_FILE"
echo ""
echo "To analyze results, run:"
echo "  python analyze_moe_search.py moe_search_results/grouped_search_*"
