export LIBFABRIC_HOME=/root/libfabric_writedata/install
export LD_LIBRARY_PATH=/usr/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=/usr/lib/aarch64-linux-gnu:$LD_LIBRARY_PATH

export TORCH_CMAKE_PREFIX_PATH=$(python3 -c "import torch; print(torch.utils.cmake_prefix_path)")
python3 -m build --wheel
python3 -m pip install ./dist/*.whl --break-system-packages --force-reinstall --no-build-isolation --no-deps
# pip install --no-build-isolation -e . --break-system-packages
