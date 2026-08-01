#!/usr/bin/env python3
"""Overlay free-run prediction vs true Lorenz state from Campaign_Trace CSVs.

CSV columns (from FreeRun dump):
  step, lt, err, locked, pred_x, pred_y, pred_z, true_x, true_y, true_z, drive_...

Usage (from repo root)::

    python examples/Lorenz/plot_freerun_overlay.py "C:\\HypercubeESN\\results\\traces\\seed21978990_orbit9333312947715283458.csv"

CSVs are written to ``{config::RESULTS_DIR}/traces/`` (absolute path; independent of
process CWD). Campaign_Trace prints the full path at the end of a run.

Writes a PNG next to the CSV (or to --out-dir) with:
  1) channel-RMS err vs Lyapunov time (+ theta line, lock shading)
  2) true vs pred x, y, z (normalized) for overlay
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib required: pip install matplotlib", file=sys.stderr)
    sys.exit(1)

THETA = 0.25  # must match config::VPT_THRESHOLD unless you override --theta


def load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data is None or data.size == 0:
        raise SystemExit(f"empty or unreadable: {path}")
    return data


def plot_one(path: Path, out_dir: Path | None, theta: float) -> None:
    d = load_csv(path)
    lt = np.asarray(d["lt"], dtype=float)
    err = np.asarray(d["err"], dtype=float)
    locked = np.asarray(d["locked"], dtype=float)
    pred = {c: np.asarray(d[f"pred_{c}"], dtype=float) for c in "xyz"}
    true = {c: np.asarray(d[f"true_{c}"], dtype=float) for c in "xyz"}

    duty = float(np.mean(locked))
    unlocks = np.where(err > theta)[0]
    vpt_lt = float(lt[unlocks[0]]) if len(unlocks) else float(lt[-1])
    vpt_x_duty = vpt_lt * duty

    fig, axes = plt.subplots(4, 1, figsize=(11, 10), sharex=True)
    fig.suptitle(
        f"{path.name}\n"
        f"duty={duty:.3f}  first err>θ at {vpt_lt:.2f} lt  "
        f"VPT×duty={vpt_x_duty:.3f}  (θ={theta})"
    )

    # Lines only (no markers). Error is raw sample-wise RMS — no smoothing.
    line_kw = dict(linestyle="-", marker="None", markevery=None)

    ax = axes[0]
    ax.plot(lt, err, color="C0", lw=1.0, label="channel-RMS err (raw)", **line_kw)
    ax.axhline(theta, color="C3", ls="--", lw=1.2, label=f"θ={theta}")
    ax.fill_between(lt, 0, theta, where=locked > 0.5, color="C2", alpha=0.15, label="locked")
    ax.set_ylabel("err")
    ax.set_ylim(0, max(1.0, float(np.nanmax(err)) * 1.05))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    for i, c in enumerate("xyz"):
        ax = axes[i + 1]
        ax.plot(lt, true[c], color="k", lw=1.1, label=f"true {c}", **line_kw)
        ax.plot(lt, pred[c], color="C1", lw=1.0, alpha=0.9, label=f"pred {c}", **line_kw)
        ax.set_ylabel(f"{c} (norm)")
        ax.legend(loc="upper right", fontsize=8)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Lyapunov time")
    fig.tight_layout()

    if out_dir is None:
        out_path = path.with_suffix(".png")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (path.stem + ".png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}  duty={duty:.3f}  first_cross={vpt_lt:.2f} lt  n={len(lt)}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("csv", nargs="+", type=Path, help="trace CSV path(s)")
    ap.add_argument(
        "-o",
        "--out-dir",
        type=Path,
        default=None,
        help="directory for PNGs (default: next to each CSV)",
    )
    ap.add_argument(
        "--theta",
        type=float,
        default=THETA,
        help=f"VPT threshold for lock shading (default {THETA})",
    )
    args = ap.parse_args()
    for p in args.csv:
        if not p.is_file():
            print(f"skip missing {p}", file=sys.stderr)
            continue
        plot_one(p, args.out_dir, args.theta)


if __name__ == "__main__":
    main()
