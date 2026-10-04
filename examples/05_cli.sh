#!/usr/bin/env bash
# Command-line workflow on plain .npy arrays (any shape; each sample is flattened).
#   bash examples/05_cli.sh
set -euo pipefail
cd "$(dirname "$0")/.."
tmp=$(mktemp -d)
python - "$tmp" <<'PY'
import sys, numpy as np
rng, d, n = np.random.default_rng(0), 16, 100_000
def t(df, n):
    return rng.standard_normal((n, d)) / np.sqrt(rng.chisquare(df, (n, 1)) / df) * np.sqrt((df - 2) / df)
np.save(f"{sys.argv[1]}/calibration.npy", t(8, n))
np.save(f"{sys.argv[1]}/reference.npy", t(8, n))
np.save(f"{sys.argv[1]}/model.npy", t(5.5, n))          # a sampler with too heavy a tail
PY
echo "== audit the raw sampler"
python -m rarecal.cli audit --calibration "$tmp/calibration.npy" --reference "$tmp/reference.npy" \
       --model "$tmp/model.npy" --components 16
echo; echo "== wrap it"
python -m rarecal.cli wrap --calibration "$tmp/calibration.npy" --model "$tmp/model.npy" \
       --out "$tmp/wrapped.npy" --components 16
echo; echo "== audit the wrapped sampler"
python -m rarecal.cli audit --calibration "$tmp/calibration.npy" --reference "$tmp/reference.npy" \
       --model "$tmp/wrapped.npy" --components 16 --json "$tmp/report.json"
rm -rf "$tmp"
