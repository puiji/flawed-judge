"""Plot RL arms: judge reward (dashed) vs. true accuracy (solid) per training step.

    uv run --with matplotlib python rl_plot.py label=outputs/rl/<run> ... -o out.png

True accuracy is format-lenient: the code check, or a \\boxed{} number equal to the gold answer.
Means are over every rollout of a step, including no-signal groups the trainer discarded.
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

COLORS = ["#2a78d6", "#eb6834", "#1baf7a"]  # reference palette slots 1, 2, and the plot.py GOLD


def per_step(run: Path) -> dict[int, tuple[float, float, float]]:
    """step -> (judge reward, strict true, lenient true) means."""
    acc = defaultdict(lambda: [0, 0.0, 0.0, 0.0])
    with open(run / "monitors/file/traces/stream/00000.jsonl") as f:
        records = [json.loads(line) for line in f]
    for d in records:
        t = d["traces"][0]
        score = (t["rewards"].get("judge_score") or {}).get("score")
        if not d["ok"] or score is None:
            continue
        reply = (
            next((n["message"]["content"] for n in reversed(t["nodes"]) if n["message"]["role"] == "assistant"), "")
            or ""
        )
        gold = d["task"]["data"]["answer"].replace(",", "")
        strict = t["metrics"].get("true_correct", 0)
        boxed = any(gold == re.sub(r"[^\d.\-]", "", s) for s in re.findall(r"boxed\{([^}]*)\}", reply))
        a = acc[d["run"]["work"]["step"]]
        a[0] += 1
        a[1] += score
        a[2] += strict
        a[3] += bool(strict or boxed)
    # Rollouts in flight when a run is stopped land on steps that never trained; drop those thin tails.
    return {s: (j / n, st / n, le / n) for s, (n, j, st, le) in sorted(acc.items()) if n >= 64}


def smooth(ys: list[float], k: int = 3) -> list[float]:
    return [sum(ys[max(0, i - k + 1) : i + 1]) / len(ys[max(0, i - k + 1) : i + 1]) for i in range(len(ys))]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="label=outputs/rl/<run>")
    p.add_argument("-o", "--out", type=Path, default=Path("rl_arms.png"))
    p.add_argument("--title", default="GRPO on GSM8K, Qwen3-0.6B, Llama-3.2-3B judge without the answer")
    args = p.parse_args()

    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=200, facecolor="white")
    for spec, color in zip(args.runs, COLORS):
        label, path = spec.rsplit("=", 1)
        rows = per_step(Path(path))
        steps = list(rows)
        judge, _, lenient = zip(*rows.values())
        if not label.startswith("control"):  # the control's reward is the code check, not the judge
            ax.plot(steps, smooth(list(judge)), color=color, lw=2, ls="--")
        ax.plot(steps, smooth(list(lenient)), color=color, lw=2, label=label)
        print(
            f"{label}: steps {steps[0]}-{steps[-1]}, judge {judge[0]:.2f}->{judge[-1]:.2f}, "
            f"true {lenient[0]:.2f}->{lenient[-1]:.2f}"
        )

    ax.plot([], [], color="#52514e", lw=2, ls="--", label="dashed = judge reward")
    ax.plot([], [], color="#52514e", lw=2, label="solid = true accuracy")
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("training step (3-step rolling mean)")
    ax.set_ylabel("mean over rollouts")
    ax.set_title(args.title, fontsize=10)
    ax.grid(alpha=0.25)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    fig.savefig(args.out)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
