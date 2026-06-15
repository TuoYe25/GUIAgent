#!/bin/bash
# macOS setup for GUI Agent benchmark & demo
set -e

echo "=== GUI Agent macOS Setup ==="

# 1. Install Python deps (skip vllm/gputil - Linux only)
pip install -r requirements.txt

# 2. Install Playwright browsers
python -m playwright install chromium

# 3. Verify torch backend
python -c "
import torch
if torch.backends.mps.is_available():
    print('MPS (Apple GPU): available')
elif torch.cuda.is_available():
    print(f'CUDA: {torch.cuda.get_device_name(0)}')
else:
    print('CPU only')
"

# 4. Electron demo deps
cd electron_demo
npm install
cd ..

echo ""
echo "=== Setup complete ==="
echo "Run benchmark:  python scripts/run_benchmarks.py --models ui-tars-1.5-7b --methods coordinate dom a11y --runs 3"
echo "Run Electron:   cd electron_demo && npm start"
