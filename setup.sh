#!/usr/bin/env bash

set -euo pipefail

# conda create -n preferthinker python=3.11
# conda activate preferthinker

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="${SCRIPT_DIR}/src/open-r1-multimodal"

cd "${PROJECT_DIR}"

python -m pip install -e ".[dev,torch,vlm]"
python -m pip install flash_attn==2.7.4.post1 --no-build-isolation
