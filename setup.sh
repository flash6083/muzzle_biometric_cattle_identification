#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "=== Creating virtual environment ==="
python3 -m venv .venv
source .venv/bin/activate

echo "=== Upgrading pip ==="
pip install --upgrade pip

echo "=== Installing dependencies ==="
pip install -r requirements.txt

echo "=== Verifying MPS support ==="
python3 -c "import torch; print('MPS available:', torch.backends.mps.is_available())"

echo "=== Verifying Ultralytics ==="
python3 -c "import ultralytics; print('Ultralytics version:', ultralytics.__version__)"

echo "=== Setup complete ==="
echo "Activate with: source .venv/bin/activate"
