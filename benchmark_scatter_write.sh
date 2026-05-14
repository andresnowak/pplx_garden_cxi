#!/usr/bin/env bash

set -euo pipefail

JOB_NAME="${JOB_NAME:-pplx-scatter-write-benchmark}"
RESERVATION="${RESERVATION:-SD-69241-apertus-1-5-0}"

read -r -a NODES_LIST <<< "${NODES_LIST:-1 2 4 8}"
read -r -a MAX_NUM_TOKENS_LIST <<< "${MAX_NUM_TOKENS_LIST:-128 4096}"

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

mkdir -p "${BENCH_LOG_DIR}" "${BENCH_JSON_DIR}"

for nodes in "${NODES_LIST[@]}"; do
    for max_num_tokens in "${MAX_NUM_TOKENS_LIST[@]}"; do
        echo "Submitting nodes=${nodes} max_num_tokens=${max_num_tokens}"
        sbatch \
            --job-name="${JOB_NAME}" \
            --dependency=singleton \
            --reservation="${RESERVATION}" \
            --output="${BENCH_LOG_DIR}/%x-%j.out" \
            --error="${BENCH_LOG_DIR}/%x-%j.err" \
            --nodes="${nodes}" \
            --export=ALL,MAX_NUM_TOKENS="${max_num_tokens}",NETS_PER_GPU="${NETS_PER_GPU}",NUM_EXPERTS="${NUM_EXPERTS}",HIDDEN_DIM="${HIDDEN_DIM}",NUM_EXPERTS_PER_TOKEN="${NUM_EXPERTS_PER_TOKEN}",IN_DTYPE="${IN_DTYPE}",NUM_WARMUP="${NUM_WARMUP}",NUM_REPEATS="${NUM_REPEATS}",OUTPUT_PREFIX="${OUTPUT_PREFIX}",BENCH_LOG_DIR="${BENCH_LOG_DIR}",BENCH_JSON_DIR="${BENCH_JSON_DIR}" \
            benchmark_scatter_write.sbatch
    done
done
