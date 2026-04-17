# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 The Vitruvian Authors
#
# Source this file to prepare a shell for running Vitruvian's sim/RL
# stack:
#
#     source scripts/env-setup.sh
#
# Why this exists
# ---------------
# JAX 0.10 (cuda12 wheels) ships its own NVIDIA libraries as pip
# packages (nvidia-cusparse-cu12, nvidia-cublas-cu12, nvidia-cudnn-cu12,
# ...). If LD_LIBRARY_PATH points at a system CUDA install — e.g.
# /usr/local/cuda — the dynamic linker finds the system copies first,
# cuSPARSE version-detection fails, and JAX silently falls back to CPU.
#
# The fix: unset LD_LIBRARY_PATH so JAX loads its bundled libs. This is
# benign for every other tool we use in this project.
#
# See docs/journal/2026-04-16.md for the diagnosis.

unset LD_LIBRARY_PATH

# Share the 8 GB card between JAX and Warp. JAX's default 75 % VRAM
# pre-allocation collides with Warp's on-demand allocator and with the
# desktop compositor's ~3 GB footprint on the RTX 4060 Ti. Disable
# pre-allocation so both allocators grow lazily.
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Activate the uv-managed venv if it exists (optional; `uv run` works
# without activation).
if [ -f .venv/bin/activate ]; then
    # shellcheck disable=SC1091
    . .venv/bin/activate
fi
