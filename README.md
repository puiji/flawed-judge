# flawed-judge

A GSM8K environment for [verifiers](https://github.com/PrimeIntellect-ai/verifiers) where the reward is an LLM judge's verdict and a code check of the real answer is logged alongside, to measure how far the two drift apart.

- Picking the best of 16 Llama-3.2-1B answers by a Llama-3.2-3B judge that doesn't see the answer gives picks that are right 37% of the time, vs 59% when the same judge is shown the gold answer (the judge rates its own picks 1.00).
- One appended string gets 98% of wrong answers past that 3B judge even when it's shown the gold answer. GPT-4.1-mini resisted every attack tried on it, and prompt hardening only moved the problem around.
- In GRPO with prime-rl (one seed per arm) the judge reward saturates in about 15 steps and true accuracy levels off below a control trained on the code check. The policy never learned injections.

Details in [RESULTS.md](RESULTS.md). Rough edges hit in verifiers and prime-cli along the way, with an upstream issue and PR, are in [ROUGH_EDGES.md](ROUGH_EDGES.md). Environment options are in the [env README](environments/flawed_judge/README.md).

```bash
uv sync
uv tool install prime && prime login   # judge calls go through Prime Inference
uv run vf-eval @ configs/eval/smoke.toml
```

MIT license.
