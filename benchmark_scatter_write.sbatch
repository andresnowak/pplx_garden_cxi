#!/usr/bin/env bash

#SBATCH --account=a139
#SBATCH --time=00:10:00
#SBATCH --job-name=pplx-scatter-write-benchmark
#SBATCH --output=bench_logs/%x-%j.out
#SBATCH --error=bench_logs/%x-%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --reservation=SD-69241-apertus-1-5-0

set -euo pipefail

export FI_CXI_ENABLE_WRITEDATA=1
export RUST_BACKTRACE=full
export MASTER_ADDR="${MASTER_ADDR:-$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)}"
export MASTER_PORT="${MASTER_PORT:-29300}"
export PPLX_LOG_COLOR=never
export DD_ENV=1

# CXI
export FI_CXI_OFLOW_BUF_SIZE=1073741824 # 1GB
export FI_CXI_RDZV_THRESHOLD=1073741824 # 1GB, don't want RDZV
export FI_CXI_DEFAULT_CQ_SIZE=131072
export FI_CXI_DEFAULT_TX_SIZE=16384 # 16k is the max
export FI_CXI_DISABLE_HOST_REGISTER=1
export FI_CXI_RX_MATCH_MODE=software # hybrid before
export FI_MR_CACHE_MONITOR=userfaultfd # This is already the default always


MAX_NUM_TOKENS="${MAX_NUM_TOKENS:?MAX_NUM_TOKENS must be set}"
NETS_PER_GPU="${NETS_PER_GPU:-1}"
NUM_EXPERTS="${NUM_EXPERTS:-256}"
HIDDEN_DIM="${HIDDEN_DIM:-2048}"
NUM_EXPERTS_PER_TOKEN="${NUM_EXPERTS_PER_TOKEN:-8}"
IN_DTYPE="${IN_DTYPE:-float16}"
NUM_WARMUP="${NUM_WARMUP:-20}"
NUM_REPEATS="${NUM_REPEATS:-30}"
OUTPUT_PREFIX="${OUTPUT_PREFIX:-scatter_write}"
BENCH_LOG_DIR="${BENCH_LOG_DIR:-bench_logs}"
BENCH_JSON_DIR="${BENCH_JSON_DIR:-log_output_scatter_write}"

# Build on a single node first, then all nodes install the pre-built wheel.
# srun \
#     --environment=apertus2-alps4-temp \
#     --mpi=pmix \
#     --network=disable_rdzv_get \
#     --nodes=1 \
#     --ntasks=1 \
#     bash -lc "
#         cd /users/anowak/open_source/pplx_garden_cxi
#         bash srun_compile.sh
#     "

srun \
    --environment=apertus2-alps4-temp \
    --mpi=pmix \
    --network=disable_rdzv_get \
    --export=FI_CXI_ENABLE_WRITEDATA,RUST_BACKTRACE,MASTER_ADDR,MASTER_PORT,SLURM_NNODES,SLURM_NODEID,PPLX_LOG_COLOR,DD_ENV,MAX_NUM_TOKENS,NETS_PER_GPU,NUM_EXPERTS,HIDDEN_DIM,NUM_EXPERTS_PER_TOKEN,IN_DTYPE,NUM_WARMUP,NUM_REPEATS,OUTPUT_PREFIX,BENCH_LOG_DIR,BENCH_JSON_DIR,FI_CXI_OFLOW_BUF_SIZE,FI_CXI_RDZV_THRESHOLD,FI_CXI_DEFAULT_CQ_SIZE,FI_CXI_DEFAULT_TX_SIZE,FI_CXI_DISABLE_HOST_REGISTER,FI_CXI_RX_MATCH_MODE,FI_MR_CACHE_MONITOR \
    bash -lc "
        cd /users/anowak/open_source/pplx_garden_cxi
        /opt/venv/bin/pip install ./dist/*.whl --force-reinstall --no-build-isolation --no-deps
        mkdir -p ${BENCH_LOG_DIR} ${BENCH_JSON_DIR}
        json_output=${BENCH_JSON_DIR}/${OUTPUT_PREFIX}_t${MAX_NUM_TOKENS}_e${NUM_EXPERTS}_h${HIDDEN_DIM}_k${NUM_EXPERTS_PER_TOKEN}_${IN_DTYPE}_world\$((SLURM_NNODES * 4)).json

        python3 -m benchmarks.bench_scatter_write \
            --world-size \$((SLURM_NNODES * 4)) \
            --nets-per-gpu ${NETS_PER_GPU} \
            --init-method=tcp://\$MASTER_ADDR:\$MASTER_PORT \
            --node-rank=\$SLURM_NODEID \
            --max-num-tokens ${MAX_NUM_TOKENS} \
            --num-experts ${NUM_EXPERTS} \
            --hidden-dim ${HIDDEN_DIM} \
            --num-experts-per-token ${NUM_EXPERTS_PER_TOKEN} \
            --in-dtype=${IN_DTYPE} \
            --num-warmup ${NUM_WARMUP} \
            --num-repeats ${NUM_REPEATS} \
            --output=\$json_output

        # Want this HIGH (low-overhead optimized writes arriving):
        cat /sys/class/cxi/cxi*/device/telemetry/C_CNTR_IXE_RX_PTL_RESTRICTED_PKT

        # Want this LOW (standard ordered RDMA — means optimized MR not being used):
        cat /sys/class/cxi/cxi*/device/telemetry/C_CNTR_IXE_RX_PTL_UNRESTRICTED_PKT

        # Want this HIGH (no-counter writes, fastest path):
        cat /sys/class/cxi/cxi*/device/telemetry/C_CNTR_LPE_PLEC_HITS

    "
