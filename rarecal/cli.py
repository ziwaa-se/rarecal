"""Command-line interface.

    rarecal audit --calibration cal.npy --reference ref.npy --model samples.npy
    rarecal wrap  --calibration cal.npy --model samples.npy --out wrapped.npy --box 0 255

Arrays are (n, ...) numpy files; each sample is flattened. `--calibration` fits
the severity functional and the calibration law, `--reference` (held out)
defines the thresholds tau_p.
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from . import WhitenedRadius, CalibrationLaw, audit, Wrapper


def _load(path):
    X = np.load(path)
    return X.reshape(len(X), -1).astype(np.float64)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="rarecal", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("audit", "wrap"):
        s = sub.add_parser(name)
        s.add_argument("--calibration", required=True, help="reference samples used to fit R and F_hat")
        s.add_argument("--model", required=True, help="model samples to audit / wrap")
        s.add_argument("--components", type=int, default=50)
    a = sub.choices["audit"]
    a.add_argument("--reference", required=True, help="held-out reference samples (define tau_p)")
    a.add_argument("--depths", type=float, nargs="+", default=[1e-2, 1e-3])
    a.add_argument("--json", help="write the report as JSON")
    w = sub.choices["wrap"]
    w.add_argument("--out", required=True)
    w.add_argument("--tail-only", type=float, default=0.99)
    w.add_argument("--box", type=float, nargs=2, default=None)
    args = ap.parse_args(argv)

    Xc = _load(args.calibration)
    sev = WhitenedRadius(args.components).fit(Xc)
    Xm = _load(args.model)
    if args.cmd == "audit":
        rep = audit(sev(Xm), sev(_load(args.reference)), depths=args.depths)
        print(rep)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(rep.to_records(), f, indent=2)
    else:
        law = CalibrationLaw.fit(sev(Xc))
        Xw, info = Wrapper(sev, law, tail_only=args.tail_only,
                           box=tuple(args.box) if args.box else None)(Xm, return_info=True)
        np.save(args.out, Xw.reshape(np.load(args.model).shape))
        print(f"moved {info['moved']} of {len(Xm)} samples "
              f"(median severity residual {info['median_residual']:.2e}); wrote {args.out}")


if __name__ == "__main__":
    main()
