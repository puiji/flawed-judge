# flawed-judge

GSM8K tasks scored by an LLM judge. The code-checked answer is logged alongside as a metric.

| Field | Default | Description |
|---|---|---|
| `split` | `test` | GSM8K split |
| `task.judge_mode` | `no_reference` | `reference` shows the judge the gold answer, `no_reference` does not |
| `task.judge.model` | verifiers default | Judge model |

Reward: `judge_score` (judge verdict, 0/1). Metrics: `reference_judge` (verdict with the gold answer), `true_correct` (code check).

```bash
uv run vf-eval @ configs/eval/smoke.toml
```
