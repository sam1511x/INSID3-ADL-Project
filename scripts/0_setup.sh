#!/usr/bin/env bash
# One-time environment setup (conda env 'insid3', PyTorch CUDA 12.6, HF access check).
set -e
cd "$(dirname "$0")/.."
if ! conda env list | grep -q "^insid3 "; then
  conda create -y -n insid3 python=3.10
fi
eval "$(conda shell.bash hook)"
conda activate insid3
pip install -r requirements.txt
pip install "transformers>=4.56" huggingface_hub gdown==5.2.0 pandas
if [ "${CRF:-0}" = "1" ]; then           # optional CRF refinement (paper's main setting)
  [ -d CRF ] || git clone https://github.com/netw0rkf10w/CRF.git
  (cd CRF && python setup.py install)
fi
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
if torch.cuda.is_available():
    print("VRAM %.1f GB, bf16 supported: %s" % (torch.cuda.get_device_properties(0).total_memory / 2**30,
                                               torch.cuda.is_bf16_supported()))
PY
if ! ls pretrain/*.pth >/dev/null 2>&1; then
  echo "No pretrain/*.pth found -> DINOv3 will be pulled from the HuggingFace Hub."
  python -c "from huggingface_hub import whoami; print('HF login OK:', whoami()['name'])" \
    || echo "!! Run 'huggingface-cli login' (token with access to facebook/dinov3-* repos)"
fi
