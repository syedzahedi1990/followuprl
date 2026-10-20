#!/usr/bin/env bash
# Install this checkout into an isolated environment. No rental or account actions.
set -euo pipefail

repo_dir="$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile="gpu"
python_bin="python3"
venv_dir=""
while (($#)); do
  case "$1" in
    --cpu) profile="cpu"; shift ;;
    --gpu) profile="gpu"; shift ;;
    --python) python_bin="${2:?--python needs an executable}"; shift 2 ;;
    --venv) venv_dir="${2:?--venv needs a path}"; shift 2 ;;
    -h|--help)
      printf '%s\n' 'Usage: bash scripts/bootstrap_vast.sh [--gpu|--cpu] [--python python3] [--venv PATH]'
      exit 0 ;;
    *) printf 'Unknown argument: %s\n' "$1" >&2; exit 2 ;;
  esac
done

if [[ -z "$venv_dir" ]]; then
  venv_dir="$repo_dir/.venv"
  [[ "$profile" != "gpu" ]] || venv_dir="$repo_dir/.venv-gpu"
fi
cd "$repo_dir"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$repo_dir/.cache/pip}"
lock_file="requirements.lock"
if [[ "$profile" == "gpu" ]]; then
  lock_file="requirements-gpu.lock"
  command -v nvidia-smi >/dev/null || { printf '%s\n' 'A visible NVIDIA GPU is required for --gpu.' >&2; exit 2; }
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
[[ -s "$lock_file" ]] || { printf 'Missing resolved dependency lock: %s\n' "$lock_file" >&2; exit 2; }
"$python_bin" - <<'PY'
import sys
if sys.version_info[:2] != (3, 12):
    raise SystemExit('This lock is validated on Python 3.12. Select a Python 3.12 runtime.')
print('Python:', sys.version.split()[0])
PY
if [[ ! -e "$venv_dir/bin/python" ]]; then
  "$python_bin" -m venv "$venv_dir"
fi
if ! "$venv_dir/bin/python" -m pip --version >/dev/null 2>&1; then
  "$venv_dir/bin/python" -m ensurepip
fi
"$venv_dir/bin/python" -m pip install --require-hashes -r "$lock_file"
"$venv_dir/bin/python" -m pip install --no-deps --no-build-isolation -e "$repo_dir"
"$venv_dir/bin/python" -m pip check
if [[ "$profile" == "gpu" ]]; then
  "$venv_dir/bin/python" - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit('CUDA is unavailable after installation; inspect the wheel/driver before continuing.')
if not torch.cuda.is_bf16_supported():
    raise SystemExit('This pilot requires BF16 support; use the declared GPU profile or separately validate a new precision profile.')
print('GPU:', torch.cuda.get_device_name())
print('VRAM GiB:', round(torch.cuda.get_device_properties(0).total_memory / 2**30, 2))
print('PyTorch/CUDA:', torch.__version__, torch.version.cuda)
PY
fi
"$venv_dir/bin/python" -m pytest -q
printf 'Installed %s profile. Python executable: %s/bin/python\n' "$profile" "$venv_dir"
printf '%s\n' 'Next: prepare data, run model preflight, and run the bounded smoke job using a local authorized configuration.'
