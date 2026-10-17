"""Best-of-n from one vf-eval run: judge score vs. true accuracy of the attempt the judge picks.

    python3 analyze.py outputs/<run-dir>

For each task, every n-subset of its attempts is scored: pick the attempt with the highest
`judge_score` (ties split evenly), and record that attempt's judge score, reference-judge
verdict and `true_correct`. Exact, no sampling. CIs are 95% normal intervals across tasks.
"""

import json
import sys
from itertools import combinations
from math import sqrt
from pathlib import Path


def load(run_dir: Path) -> list[list[tuple[float, float, float]]]:
    tasks: dict[int, list] = {}
    for line in open(run_dir / "traces.jsonl"):  # one line per attempt
        episode = json.loads(line)
        for t in episode["traces"]:
            if t["ok"]:
                tasks.setdefault(episode["task"]["data"]["idx"], []).append(
                    (t["rewards"]["judge_score"]["score"], t["metrics"]["reference_judge"], t["metrics"]["true_correct"])
                )
    return list(tasks.values())


def best_of(attempts: list[tuple[float, ...]], n: int) -> tuple[float, ...]:
    """Expected (judge, reference, true) of the judge's pick, over all n-subsets."""
    totals, subsets = [0.0] * 3, 0
    for subset in combinations(attempts, n):
        top = max(a[0] for a in subset)
        picks = [a for a in subset if a[0] == top]
        for i in range(3):
            totals[i] += sum(p[i] for p in picks) / len(picks)
        subsets += 1
    return tuple(t / subsets for t in totals)


def mean_ci(xs: list[float]) -> str:
    m = sum(xs) / len(xs)
    sd = sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1)) if len(xs) > 1 else 0.0
    return f"{m:.3f} ± {1.96 * sd / sqrt(len(xs)):.3f}"


def main() -> None:
    tasks = load(Path(sys.argv[1]))
    r = min(len(a) for a in tasks)
    print(f"{len(tasks)} tasks, {r}+ ok attempts each\n")
    print(f"{'n':>3}  {'judge (no ref)':>15}  {'judge (ref)':>15}  {'true accuracy':>15}  {'gap':>15}")
    n = 1
    while n <= r:
        rows = [best_of(a, n) for a in tasks]
        judge, ref, true = ([row[i] for row in rows] for i in range(3))
        gap = [j - t for j, t in zip(judge, true)]
        print(f"{n:>3}  {mean_ci(judge):>15}  {mean_ci(ref):>15}  {mean_ci(true):>15}  {mean_ci(gap):>15}")
        n *= 2


if __name__ == "__main__":
    assert best_of([(1, 0, 0), (0, 1, 1)], 2) == (1.0, 0.0, 0.0)  # judge picks the wrong one
    assert best_of([(1, 1, 1), (1, 0, 0)], 2) == (1.0, 0.5, 0.5)  # ties split evenly
    assert best_of([(1, 1, 1), (0, 0, 0)], 1) == (0.5, 0.5, 0.5)
    main()
