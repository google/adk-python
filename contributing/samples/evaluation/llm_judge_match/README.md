# LLM-judged response match

## Overview

Evaluates the shared home-automation agent with `final_response_match_v2`, a
criterion that uses a judge model to decide whether the agent's final answer
is *semantically* equivalent to a reference response. Because the judge reasons
about meaning rather than word overlap, it tolerates phrasing and formatting
differences that `response_match_score` (ROUGE-1) would penalize, for example
"The temperature in the Kitchen is currently 24°C." vs. the reference "It's
currently 24 degrees Celsius in the Kitchen."

This criterion needs a model credential for the judge (a Gemini API key or
Vertex), in addition to the credential used for the agent's own inference.

## Sample Inputs

The eval set (`home_automation.evalset.json`) contains one single-turn case:

- `How warm is the Kitchen right now?`

The reference answer is deliberately phrased differently from how the agent is
likely to respond, so ROUGE-1 would score low while the semantic judge passes.

## How To

Run the sample from the workspace root:

```bash
adk eval contributing/samples/evaluation/home_automation_agent \
    contributing/samples/evaluation/llm_judge_match/home_automation.evalset.json \
    --config_file_path contributing/samples/evaluation/llm_judge_match/eval_config.json \
    --print_detailed_results
```

`adk eval` takes the agent folder and the eval-set file as two separate
arguments, so this folder holds only eval data (`home_automation.evalset.json`),
the criteria config (`eval_config.json`), and this README, with no agent code.

### `judge_model_options`

`final_response_match_v2` is configured through `judge_model_options` in
`eval_config.json`:

- `judge_model`: the model that acts as the judge. It is resolved through the
  standard ADK model registry, so it is a normal model name (here
  `gemini-2.5-flash`). The judge is a separate model from the one the agent uses
  for its own inference.
- `num_samples`: how many independent judgements to request from the judge
  model (here `5`). The criterion takes a majority vote across those samples
  and converts the fraction of "equivalent" votes into the score, which reduces
  the impact of any single noisy judgement.

The `threshold` is `0.8`, so at least a strong majority of the judge samples must
find the responses equivalent for the case to pass.

### Semantic vs. lexical matching

Use `final_response_match_v2` when a correct answer can legitimately be worded or
formatted many different ways and you care about *meaning*, not exact wording:
paraphrases, reordered clauses, "24°C" vs. "24 degrees Celsius", extra polite
framing, and so on. A lexical metric like `response_match_score` (ROUGE-1, used
in the `basic_criteria` sample) only measures word overlap, so it would penalize
these harmless rephrasings and force you to lower the threshold until it no
longer distinguishes right answers from wrong ones. The trade-off is that the
LLM judge requires a model call per sample (cost and latency) and, being
model-based, can vary slightly between runs; `response_match_score` is fully
local and deterministic. Reach for the semantic judge when meaning matters more
than phrasing, and keep the lexical metric when you need cheap, deterministic
scoring.

## Related Guides

- Evaluation overview: https://adk.dev/evaluate/
- Evaluation criteria reference: https://adk.dev/evaluate/criteria/
