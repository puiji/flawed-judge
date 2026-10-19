# flawed-judge

A GSM8K environment for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) where the reward is an LLM judge's verdict and a code check of the real answer is logged alongside, to measure how far the two drift apart.

- Best-of-16 by a Llama-3.2-3B judge without the answer scores 1.00 while true accuracy is 0.37.
- One appended line gets 98% of wrong answers past the same judge even when it's shown the gold answer.
- Under RL the judge reward saturates in about 15 steps and true accuracy stalls below a control trained on the real check.

Details in [RESULTS.md](RESULTS.md).

```bash
uv sync
prime login
uv run vf-eval @ configs/eval/smoke.toml
```

MIT
