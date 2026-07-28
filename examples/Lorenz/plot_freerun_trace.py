#!/usr/bin/env python3
"""Plot Lorenz free-run CSV traces from Lorenz.exe --trace.

CSV columns (from FreeRun dump):
  step, lt, err, locked, pred_x/y/z, true_x/y/z, past_x/y/z, past_xz

Usage (from repo root)::

    python examples/Lorenz/plot_freerun_trace.py examples/Lorenz/traces/seed*_orbit*.csv

Shows:
  1) channel-RMS err vs Lyapunov time, with VPT_THRESHOLD=0.3 and lock shading
  2) true vs predicted x (normalized) — sanity for "stuck / too good"

Something is off if: err is flat near zero while past is non-informative, or
true and pred are identical for long stretches after teacher-forcing should end.
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

THETA = 0.3  # must match config::VPT_THRESHOLD


def load_csv(path: Path) -> np.ndarray:
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data is None or data.size == 0:
        raise SystemExit(f"empty or unreadable: {path}")
    return data


def plot_one(path: Path, out_dir: Path | None) -> None:
    d = load_csv(path)
    lt = np.asarray(d["lt"], dtype=float)
    err = np.asarray(d["err"], dtype=float)
    locked = np.asarray(d["locked"], dtype=float)
    pred_x = np.asarray(d["pred_x"], dtype=float)
    true_x = np.asarray(d["true_x"], dtype=float)
    past_x = np.asarray(d["past_x"], dtype=float)

    duty = float(np.mean(locked))
    # first unlock (VPT-style): first step with err > theta
    unlocks = np.where(err > THETA)[0]
    vpt_lt = float(lt[unlocks[0]]) if len(unlocks) else float(lt[-1])

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.suptitle(
        f"{path.name}\n"
        f"duty={duty:.3f}  first_err>θ at {vpt_lt:.2f} lt  (θ={THETA})"
    )

    ax = axes[0]
    ax.plot(lt, err, color="C0", lw=1.0, label="channel-RMS err")
    ax.axhline(THETA, color="C3", ls="--", lw=1.2, label=f"θ={THETA}")
    ax.fill_between(lt, 0, THETA, where=locked > 0.5, color="C2", alpha=0.15, label="locked")
    ax.set_ylabel("err")
    ax.set_ylim(0, max(1.0, float(np.nanmax(err)) * 1.05))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    ax.plot(lt, true_x, color="k", lw=1.0, label="true x")
    ax.plot(lt, pred_x, color="C1", lw=1.0, alpha=0.85, label="pred x")
    ax.set_ylabel("x (norm)")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)

    ax = axes[2]
    ax.plot(lt, past_x, color="C4", lw=1.0, label="past_x (input drive)")
    ax.set_xlabel("Lyapunov time")
    ax.set_ylabel("past_x")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.3)
    # Under FORWARD_ONLY this should be ~0; if not, dump/arm mismatch.
    if np.nanmax(np.abs(past_x)) < 1e-6:
        ax.set_title("past_x ≈ 0 (FORWARD_ONLY or zeroed drive)", fontsize=9)
    else:
        ax.set_title("past_x nonzero (Janus real past)", fontsize=9)

    fig.tight_layout()
    if out_dir is None:
        out_path = path.with_suffix(".png")
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (path.stem + ".png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"wrote {out_path}  duty={duty:.3f}  first_cross={vpt_lt:.2f} lt")


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
    args = ap.parse_args()
    for p in args.csv:
        if not p.is_file():
            print(f"skip missing {p}", file=sys.stderr)
            continue
        plot_one(p, args.out_dir)


if __name__ == "__main__":
    main()
