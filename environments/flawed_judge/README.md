# flawed-judge

GSM8K tasks scored by an LLM judge. The code-checked answer is logged alongside as a metric.

| Field | Default | Description |
|---|---|---|
| `split` | `test` | GSM8K split |
| `task.judge_mode` | `no_reference` | `reference` shows the judge the gold answer, `no_reference` does not, `solve_first` has the judge solve the task and use its own answer as the reference |
| `task.harden` | `false` | Tell the judge to ignore instructions and verification claims inside the response |
| `task.reward` | `judge` | `judge` trains on the judge verdict, `true_correct` on the code check (RL control) |
| `task.judge.model` | verifiers default | Judge model |

Reward: `judge_score` (judge verdict, 0/1). Metrics: `reference_judge` (verdict with the gold answer), `true_correct` (code check).

```bash
uv run vf-eval @ configs/eval/smoke.toml
```
