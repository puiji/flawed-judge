# flawed-judge

A GSM8K environment for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) where the reward is an LLM judge's verdict and a code check of the real answer is logged alongside, to measure how far the two drift apart.

- Best-of-16 by a Llama-3.2-3B judge without the answer scores 1.00 on Llama-3.2-1B's answers while true accuracy is 0.37.
- One appended string gets 98% of wrong answers past the same judge even when it's shown the gold answer.
- Under RL (one seed per arm) the judge reward saturates in about 15 steps and true accuracy levels off below a control trained on the real check.

Details in [RESULTS.md](RESULTS.md). Rough edges hit in verifiers and prime-cli along the way, with an upstream issue and PR, are in [ROUGH_EDGES.md](ROUGH_EDGES.md). Environment options are in the [env README](environments/flawed_judge/README.md).

```bash
uv sync
uv tool install prime && prime login   # judge calls go through Prime Inference
uv run vf-eval @ configs/eval/smoke.toml
```

MIT license.
