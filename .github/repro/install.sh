#!/usr/bin/env bash
# Install unsloth-zoo at $1 plus the MLX stack the reporters' Unsloth Desktop runs (see the issue thread).
set -euo pipefail
ref="$1"
sysctl -n machdep.cpu.brand_string
sysctl -n hw.memsize | awk '{print $1/1073741824 " GB"}'
sw_vers -productVersion
python -m pip install --upgrade pip
for attempt in 1 2 3; do
  pip install "unsloth_zoo @ git+https://github.com/unslothai/unsloth-zoo@${ref}" && break
  [ "$attempt" -eq 3 ] && exit 1
  sleep $((5 * attempt))
done
# mlx-vlm 0.7.9 declares a newer transformers, but Desktop runs it on 5.5.0: pin both without the resolver.
pip install --no-deps 'mlx==0.32.4' 'mlx-vlm==0.7.9' 'mlx-lm==0.32.2' 'transformers==5.5.0' 'tokenizers==0.22.2'
python -c "import mlx_lm, mlx_vlm, transformers, unsloth_zoo.mlx.loader as L; print('imports ok; fp32 pooler fix present:', hasattr(L, '_fix_gemma4_vision_pooler_fp32'))"
pip list | grep -iE '^(mlx|transformers|tokenizers|unsloth)'
