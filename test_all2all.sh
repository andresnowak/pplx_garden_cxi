set -euo pipefail

export NODE_RANK=$SLURM_NODEID
export NUM_NODES=$SLURM_NNODES
export MASTER_IP="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)" &&

export RUST_BACKTRACE=full
export FI_CXI_ENABLE_WRITEDATA=1
export PPLX_LOG_COLOR=never
export DD_ENV=1
export PPLX_TEST_DEBUG=1
export PPLX_TEST_DEBUG_RECV_BUFFER=1
export PPLX_TEST_DEBUG_COMBINE_WORKER=1
export PPLX_TEST_DEBUG_COMBINE_RECV_KERNEL=1

nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv
nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill # This is necessary because it seems if there are left over processes that didn't get killed when a test failed, then they will always cause subsequent tests to fail randomly
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv

# python3 -m pytest -sv -vv -rs tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all "$@"

# python3 -m pytest -sv -vv -rs tests/p2p_all_to_all/test_p2p_all_to_all.py -k "^TP2-NIC1-BF16$" "$@"
# python3 -m pytest -sv -vv -rs "tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all[TP4-DP2-NIC1-BF16-T1024]" "$@"
# python3 -m pytest -sv -vv -rs "tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all[TP4-DP2-NIC1-BF16-T1024]" "$@"
# python3 -m pytest -sv -vv -rs "tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all[TP4-DP2-NIC1-BF16]" "$@"
python3 -m pytest -sv -vv -rs "tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all[TP4-DP2-NVL2-T256]" "$@"


# python3 -m pytest -sv -vv -rs tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all_moe_roundtrip "$@"
