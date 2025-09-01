#!/usr/bin/env bash
set -Eeuo pipefail

# Usage:
#   ./rebuild_mamba2_fused.sh [env-name] [--fast] [--reinstall] [--no-test]
# Examples:
#   ./rebuild_mamba2_fused.sh
#   ./rebuild_mamba2_fused.sh mamba --fast
#   ./rebuild_mamba2_fused.sh mamba --fast --reinstall --no-test

ENV_NAME="${1:-mamba}"

# flags
FAST=0
REINSTALL=0
RUN_TEST=1
shift || true
for arg in "$@"; do
  case "$arg" in
    --fast) FAST=1 ;;
    --reinstall) REINSTALL=1 ;;
    --no-test) RUN_TEST=0 ;;
    *) ;;
  esac
done

trap 'echo "[x] Error on line $LINENO. Aborting." >&2' ERR
cd "$(dirname "$0")"

# --- Minimal layout check (kept even in fast mode) ---
[[ -f "setup.py" ]] || { echo "[x] setup.py not found here: $(pwd)"; exit 1; }

if (( FAST == 0 )); then
  echo "[*] Verifying project layout..."
  [[ -f "mamba2_fused.cpp" ]] || { echo "[x] mamba2_fused.cpp not found"; exit 1; }
fi

# --- Conda activation (skip in FAST if you're already in env) ---
if (( FAST == 0 )); then
  echo "[*] Locating & activating conda env: $ENV_NAME"
  if ! command -v conda >/dev/null 2>&1; then
    echo "[x] 'conda' not found; install or add to PATH." >&2; exit 1
  fi
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
fi

# --- Optional sanity checks (skip in FAST) ---
if (( FAST == 0 )); then
  echo "[*] Checking Python & PyTorch..."
  python - <<'PY'
import sys
print("python:", sys.version.split()[0])
import torch
print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())
PY
fi

# --- Build speed-ups ---
# Use Ninja + parallel compilation
export USE_NINJA=1
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# Select only your current GPU arch unless already set (speeds up nvcc)
if [[ -z "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  DETECTED_ARCH="$(python - <<'PY'
import torch, sys
print("%d.%d"%(torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (8,9)))
PY
)"
  export TORCH_CUDA_ARCH_LIST="${DETECTED_ARCH}+PTX"
fi
echo "[*] TORCH_CUDA_ARCH_LIST=${TORCH_CUDA_ARCH_LIST}"

# --- Clean (cheap in FAST mode) ---
if (( FAST )); then
  echo "[*] FAST: skipping full clean (incremental build)"
else
  echo "[*] Cleaning old build artifacts..."
  rm -rf build mamba2_fused*.egg-info mamba2_fused*.so
fi

# --- Rebuild in place (quiet) ---
echo "[*] Building extension (in-place, Ninja, MAX_JOBS=${MAX_JOBS})..."
python setup.py -q build_ext --inplace

# --- (Optional) Reinstall editable ---
if (( REINSTALL )); then
  echo "[*] Reinstalling editable package..."
  python -m pip install -q -e .
else
  echo "[*] Skipping reinstall (use --reinstall to force)"
fi

# --- Quick smoke test (skip with --no-test) ---
if (( RUN_TEST )); then
  echo "[*] Running smoke test..."
  python - <<'PY'
import importlib
import sys
try:
    import torch
    import mamba2_fused
    print("✓ mamba2_fused imported successfully")
    print("loaded from:", mamba2_fused.__file__)
except Exception as e:
    print("✗ import failed:", e)
    sys.exit(1)
PY
else
  echo "[*] Skipping smoke test (--no-test)."
fi

echo "[✓] Done (FAST=${FAST}, REINSTALL=${REINSTALL})."
