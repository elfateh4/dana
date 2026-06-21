#!/usr/bin/env bash
set -euo pipefail

# Kaggle setup — run locally after pushing repo to GitHub

chmod 600 ~/.kaggle/kaggle.json

echo "=== Creating benchmark instances dataset ==="
kaggle datasets create -p dana/data/instances --dir-mode zip

echo "=== Pushing training kernel ==="
kaggle kernels push -p kaggle/train

echo "=== Pushing evaluation kernel ==="
kaggle kernels push -p kaggle/eval

echo "=== Done ==="
echo "Monitor: https://kaggle.com/elfateh/kernels"
