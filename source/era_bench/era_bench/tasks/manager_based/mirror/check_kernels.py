"""Run this script inside the Apptainer container to check Warp kernel compilation
without launching Isaac Sim:

    python check_kernels.py
"""

import warp as wp

wp.init()

# Importing the module registers all @wp.kernel and @wp.func definitions.
from mirror_state_machine import mirror_state_machine_kernel  # noqa: E402

# force_load() triggers JIT compilation of every kernel in the module immediately.
mirror_state_machine_kernel.module.load(device="cpu")

print("All kernels compiled successfully.")
