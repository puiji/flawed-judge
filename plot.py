"""Plot best-of-n: each judge's own score (dashed) vs. true accuracy of its pick (solid).

    uv run --with matplotlib python plot.py outputs/<run> [label=outputs/<replay> ...] -o out.png

The source run's judge is labeled `no_reference`. Each replay adds one judge. The gold-answer
pick and oracle pass@n come from the source run. Uses analyze.py's exact best-of-n.
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from analyze import best_of, load

NS = [1, 2, 4, 8, 16]
COLORS = ["#2a78d6", "#eb6834", "#6250d6"]  # one per judge, reference palette slots 1, 2, 7
GOLD = "#1baf7a"


def curve(tasks, n_values=NS):
    """Mean (judge, true) of the judge's pick across tasks, per n."""
    out = []
    for n in n_values:
        rows = [best_of(a, min(n, len(a))) for a in tasks]
        out.append((sum(r[0] for r in rows) / len(rows), sum(r[2] for r in rows) / len(rows)))
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("run", type=Path)
    p.add_argument("replays", nargs="*", help="label=outputs/<replay-dir>")
    p.add_argument("-o", "--out", type=Path, default=Path("best_of_n.png"))
    p.add_argument("--title", default="Llama-3.2-1B on GSM8K (50 tasks × 16 attempts)")
    args = p.parse_args()

    base = load(args.run)
    judges = [("no_reference (Llama-3B)", base)]
    for spec in args.replays:
        label, path = spec.split("=", 1)
        judges.append((label, load(args.run, Path(path))))

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200, facecolor="white")
    for (label, tasks), color in zip(judges, COLORS):
        judge, true = zip(*curve(tasks))
        ax.plot(NS, judge, color=color, lw=2, ls="--", marker="o", ms=4)
        ax.plot(NS, true, color=color, lw=2, marker="o", ms=4, label=label)
    gold = [[(r, r, t) for (_, r, t) in a] for a in base]
    oracle = [[(t, t, t) for (_, _, t) in a] for a in base]
    ax.plot(NS, [t for _, t in curve(gold)], color=GOLD, lw=2, marker="o", ms=4,
            label="gold-answer judge pick")
    ax.plot(NS, [t for _, t in curve(oracle)], color="#8a8985", lw=1.5, ls=":", label="oracle pass@n")

    ax.plot([], [], color="#52514e", lw=2, ls="--", label="dashed = judge's own score")
    ax.plot([], [], color="#52514e", lw=2, label="solid = true accuracy of its pick")
    ax.set_xscale("log", base=2)
    ax.set_xticks(NS, [str(n) for n in NS])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("n attempts per task (best-of-n by judge score)")
    ax.set_ylabel("score")
    ax.set_title(args.title, loc="left", fontsize=11)
    ax.grid(axis="y", color="#e6e5e0", lw=0.8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(fontsize=8, frameon=False, loc="center left", bbox_to_anchor=(1.12, 0.5))
    fig.savefig(args.out, bbox_inches="tight", facecolor="white")
    print(args.out)


if __name__ == "__main__":
    main()
