#!/usr/bin/env bash
set -Eeuo pipefail

ENV_NAME="${1:-mamba}"   # allow override: ./rebuild_mamba2_fused.sh <env-name>
RUN_TEST="1"
if [[ "${2:-}" == "--no-test" ]]; then RUN_TEST="0"; fi

# Pretty error reporting
trap 'echo "[x] Error on line $LINENO. Aborting." >&2' ERR

# Move to script directory
cd "$(dirname "$0")"

echo "[*] Verifying project layout..."
[[ -f "setup.py" ]] || { echo "[x] setup.py not found here: $(pwd)"; exit 1; }
[[ -f "mamba2_fused.cpp" ]] || { echo "[x] mamba2_fused.cpp not found"; exit 1; }
# If CUDA source exists, we'll later check for nvcc
HAS_CU="0"; [[ -f "mamba2_fused_cuda.cu" ]] && HAS_CU="1"

echo "[*] Locating conda..."
if ! command -v conda >/dev/null 2>&1; then
  echo "[x] 'conda' command not found in PATH. Make sure conda/mamba is installed and exported." >&2
  exit 1
fi

# Activate the desired env
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  echo "[x] Conda environment '$ENV_NAME' not found." >&2
  conda env list
  exit 1
fi

echo "[*] Activating conda env: $ENV_NAME"
conda activate "$ENV_NAME"

# Sanity checks inside env
echo "[*] Checking Python & PyTorch..."
python - <<'PY'
import sys
print("python:", sys.version.split()[0])
try:
    import torch
    print("torch:", torch.__version__, "cuda:", torch.version.cuda, "is_available:", torch.cuda.is_available())
except Exception as e:
    print("ERROR: torch not importable:", e); raise
PY

if [[ "$HAS_CU" == "1" ]]; then
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "[!] WARNING: nvcc not found but CUDA source (.cu) exists. If your build needs nvcc, install CUDA toolkit or add nvcc to PATH." >&2
  fi
fi

echo "[*] Cleaning old build artifacts (scoped)..."
rm -rf ./build
rm -rf ./mamba2_fused*.egg-info
rm -f  ./mamba2_fused*.so

echo "[*] Rebuilding extension in-place..."
python setup.py build_ext --inplace

echo "[*] Reinstalling editable package..."
pip install -e .

if [[ "$RUN_TEST" == "1" ]]; then
  echo "[*] Running quick smoke test..."
  python - <<'PY'
import torch
import mamba2_fused as mf

assert torch.cuda.is_available(), "CUDA not available (torch.cuda.is_available() is False)"
# a = torch.randn(2, 3, device="cuda", dtype=torch.float32)
# b = torch.randn(2, 3, device="cuda", dtype=torch.float32)
# out = mf.add(a, b)
# assert torch.allclose(out, a + b, rtol=1e-6, atol=1e-6), "Output mismatch vs torch add"
# print("Smoke test: OK")
PY
else
  echo "[*] Skipping smoke test (--no-test)."
fi

echo "[✓] Done. Env: '$ENV_NAME'  Dir: '$(pwd)'"
