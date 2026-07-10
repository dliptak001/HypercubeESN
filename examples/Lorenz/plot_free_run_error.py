"""Plot free-run channel-RMS error from a logged Lorenz free-run (every 25 steps)."""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Free-run log (every 25 steps) — seed 13649419, input_scaling=0.005, feedback=0.04
# (steps, lyapunov_times, err)
ROWS = [
    (25, 0.45, 0.006222),
    (50, 0.91, 0.041587),
    (75, 1.36, 0.030075),
    (100, 1.81, 0.037214),
    (125, 2.26, 0.171733),
    (150, 2.72, 0.043245),
    (175, 3.17, 0.121177),
    (200, 3.62, 0.034528),
    (225, 4.08, 0.031347),
    (250, 4.53, 0.055617),
    (275, 4.98, 0.073515),
    (300, 5.43, 0.058062),
    (325, 5.89, 0.136049),
    (350, 6.34, 0.548223),
    (375, 6.79, 0.216012),
    (400, 7.24, 0.345100),
    (425, 7.70, 0.750539),
    (450, 8.15, 0.273842),
    (475, 8.60, 0.403227),
    (500, 9.06, 0.865397),
    (525, 9.51, 0.143702),
    (550, 9.96, 0.536251),
    (575, 10.41, 0.490088),
    (600, 10.87, 0.107468),
    (625, 11.32, 0.456046),
    (650, 11.77, 0.709024),
    (675, 12.23, 0.245702),
    (700, 12.68, 0.607831),
    (725, 13.13, 0.643530),
    (750, 13.58, 0.335822),
    (775, 14.04, 0.425607),
    (800, 14.49, 0.275635),
    (825, 14.94, 0.203928),
    (850, 15.40, 0.725519),
    (875, 15.85, 0.404069),
    (900, 16.30, 0.481475),
    (925, 16.75, 0.462701),
    (950, 17.21, 0.363525),
    (975, 17.66, 0.955713),
    (1000, 18.11, 0.050125),
    (1025, 18.56, 0.052622),
    (1050, 19.02, 0.077100),
    (1075, 19.47, 0.368900),
    (1100, 19.92, 0.719322),
    (1125, 20.38, 0.414017),
    (1150, 20.83, 0.431919),
    (1175, 21.28, 0.208494),
    (1200, 21.73, 0.567313),
    (1225, 22.19, 0.498956),
    (1250, 22.64, 0.275665),
    (1275, 23.09, 0.487800),
    (1300, 23.55, 0.555814),
    (1325, 24.00, 0.035691),
    (1350, 24.45, 0.095183),
    (1375, 24.90, 0.203243),
    (1400, 25.36, 0.618120),
    (1425, 25.81, 0.570480),
    (1450, 26.26, 0.529327),
    (1475, 26.72, 0.153914),
    (1500, 27.17, 0.128160),
    (1525, 27.62, 0.718123),
    (1550, 28.07, 0.135346),
    (1575, 28.53, 0.393133),
    (1600, 28.98, 0.788199),
    (1625, 29.43, 0.232545),
    (1650, 29.88, 0.459498),
    (1675, 30.34, 0.604092),
    (1700, 30.79, 0.369735),
    (1725, 31.24, 0.621437),
    (1750, 31.70, 0.308435),
    (1775, 32.15, 0.180776),
    (1800, 32.60, 0.555923),
    (1825, 33.05, 0.295736),
    (1850, 33.51, 0.284707),
    (1875, 33.96, 0.454665),
    (1900, 34.41, 0.062427),
    (1925, 34.87, 0.062825),
    (1950, 35.32, 0.524620),
    (1975, 35.77, 0.324912),
    (2000, 36.22, 0.201762),
]

VPT_STEPS = 347
VPT_LT = 6.28
VPT_THRESH = 0.30
STEPS_PER_LT = 1.0 / (0.9056 * 0.02)  # config::LYAPUNOV_EXPONENT * DT


def main() -> None:
    lt = np.array([r[1] for r in ROWS], dtype=float)
    err = np.array([r[2] for r in ROWS], dtype=float)
    under = err < VPT_THRESH

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=140)
    ax.plot(
        lt,
        err,
        color="#1f77b4",
        lw=1.4,
        marker="o",
        ms=3.2,
        label="channel-RMS error (every 25 steps)",
    )
    ax.axhline(
        VPT_THRESH,
        color="#d62728",
        ls="--",
        lw=1.2,
        label=f"VPT threshold = {VPT_THRESH:.2f}",
    )
    ax.axvline(
        VPT_LT,
        color="#2ca02c",
        ls=":",
        lw=1.4,
        label=f"first-crossing VPT = {VPT_STEPS} steps ({VPT_LT:.2f} λt)",
    )
    ax.fill_between(lt, 0, err, where=under, color="#1f77b4", alpha=0.12)

    ax.set_xlabel("Lyapunov time (λt)")
    ax.set_ylabel("Free-run error (channel-RMS, normalized)")
    ax.set_title(
        "Lorenz free-run error — seed 13649419, input_scaling=0.005, feedback=0.04"
    )
    ax.set_xlim(0, float(lt[-1]) * 1.01)
    ax.set_ylim(0, max(1.05, float(err.max()) * 1.05))
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8.5)

    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    step_ticks = np.array([0, 500, 1000, 1500, 2000], dtype=float)
    ax2.set_xticks(step_ticks / STEPS_PER_LT)
    ax2.set_xticklabels([f"{int(s)}" for s in step_ticks])
    ax2.set_xlabel("generative step")

    fig.tight_layout()
    out = Path(__file__).with_name("free_run_error_seed13649419.png")
    fig.savefig(out, bbox_inches="tight")
    print(f"wrote {out}")
    print(f"points={len(err)}  max_err={err.max():.3f}  frac_under_threshold={under.mean():.1%}")


if __name__ == "__main__":
    main()
