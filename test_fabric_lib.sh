set -euo pipefail

export NODE_RANK=$SLURM_NODEID
export NUM_NODES=$SLURM_NNODES
export MASTER_IP="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)" &&

export RUST_BACKTRACE=full
export FI_CXI_ENABLE_WRITEDATA=1
export PPLX_LOG_COLOR=never
export DD_ENV=1
export PPLX_TEST_NETS_PER_GPU=1
export FI_LOG_LEVEL=debug
export FI_LOG_PROV=cxi
export FI_LOG_SUBSYS=mr
export RUST_LOG=debug

export PYTHONPATH="$(pwd):${PYTHONPATH:-}"

# python3 -m pytest -sv -vv -rs tests/fabric_lib "$@"
# python3 -m pytest -sv -vv -rs tests/fabric_lib/test_handle.py "$@"
# python3 -m pytest -sv -vv -rs tests/fabric_lib/test_transfer_engine.py::my_simple_write_cpu"$@"
# python3 -m pytest -sv -vv -rs tests/fabric_lib/test_transfer_engine.py "$@"
cd tests/fabric_lib
export PYTORCH_NO_CUDA_MEMORY_CACHING=0
python test_transfer_engine.py

# RUST_LOG=debug cargo run -p fabric-debug -- --register-only 0 1
# RUST_LOG=debug cargo run -p fabric-debug -- --register-only 2 1

# (
#   export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
#   cd tests/fabric_lib
#   RUST_LOG=debug cargo run -p fabric-debug -- --register-only-wait 30 0,1 1
# ) &
# (
#   export PYTHONPATH="$(pwd):${PYTHONPATH:-}"
#   cd tests/fabric_lib
#   RUST_LOG=debug cargo run -p fabric-debug -- --register-only-wait 30 2,3 1
# ) &
# wait


# FI_MR_CACHE_MAX_COUNT=0 RUST_LOG=debug cargo run -p fabric-debug -- --register-only 0 1
# FI_CXI_DISABLE_HMEM_DEV_REGISTER=1 RUST_LOG=debug cargo run -p fabric-debug -- --register-only 0 1
# FI_CXI_FORCE_DEV_REG_COPY=1 RUST_LOG=debug cargo run -p fabric-debug -- --register-only 0 1
# FI_HMEM_DISABLE_P2P=1 RUST_LOG=debug cargo run -p fabric-debug -- --register-only 0 1
