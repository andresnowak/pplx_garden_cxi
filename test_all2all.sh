set -euo pipefail

export NODE_RANK=$SLURM_NODEID
export NUM_NODES=$SLURM_NNODES
export MASTER_IP="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)" &&

export RUST_BACKTRACE=full
export FI_CXI_ENABLE_WRITEDATA=1
export PPLX_LOG_COLOR=never
export DD_ENV=1

# python3 -m pytest -sv -vv -rs tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all "$@"

# python3 -m pytest -sv -vv -rs tests/p2p_all_to_all/test_p2p_all_to_all.py -k "^TP2-NIC1-BF16$" "$@"
# python3 -m pytest -sv -vv -rs "tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all[TP4-DP2-NIC1-BF16-T1024]" "$@"
# python3 -m pytest -sv -vv -rs "tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all[TP4-DP2-NIC1-BF16-T1024]" "$@"
# python3 -m pytest -sv -vv -rs "tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all[TP4-DP2-NIC1-BF16]" "$@"
python3 -m pytest -sv -vv -rs "tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all[TP4-DP2-NVL2]" "$@"


# python3 -m pytest -sv -vv -rs tests/p2p_all_to_all/test_p2p_all_to_all.py::test_p2p_all_to_all_moe_roundtrip "$@"