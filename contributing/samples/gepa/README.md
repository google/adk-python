# Example: optimizing an ADK agent with Genetic-Pareto

This directory contains an example demonstrating how to use the Agent
Development Kit (ADK) to run and optimize an LLM-based agent in a simulated
environment with the Genetic-Pareto prompt optimization algorithm
([GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning](https://arxiv.org/abs/2507.19457))
on benchmarks like Tau-bench.

## Goal

The goal of this demo is to take an agent with a simple, underperforming prompt
and automatically improve it using GEPA, increasing the agent's reliability on a
customer support task.

## Examples

### Tau-Bench Retail Environment

We use the `'retail'` environment from
[Tau-bench](https://github.com/sierra-research/tau-bench), a benchmark designed
to test agents in realistic, conversational scenarios involving tool use and
adherence to policies. In this environment, our agent acts as a customer
support agent for an online store. It needs to use a set of tools (like
`check_order_status`, `issue_refund`, etc.) to help a simulated user resolve
their issues, while following specific support policies (e.g., only refunding
orders less than 30 days old). The agent is built with ADK using a standard
tool-calling strategy. It receives the conversation history and a list of
available tools, and it must decide whether to respond to the user or call a
tool.

The easiest way to run this demo is through the provided Colab notebook:
[`gepa_tau_bench.ipynb`](https://colab.research.google.com/github/google/adk-python/blob/main/contributing/samples/gepa/gepa_tau_bench.ipynb).

### Improving a voter Agent's PII filtering ability

This demo notebook ([`voter_agent/gepa.ipynb`](https://colab.research.google.com/github/google/adk-python/blob/main/contributing/samples/gepa/voter_agent/gepa.ipynb)) walks you through optimizing an AI
agent's prompt using the Genetic-Pareto (GEPA) algorithm. We'll use the Google
Agent Development Kit (ADK) to build and evaluate a "Vote Taker" agent designed
to collect audience votes while filtering sensitive information.


## GEPA Overview

**GEPA (Genetic-Pareto)** is a prompt optimization algorithm that learns from
trial and error, using LLM-based reflection to understand failures and guide
prompt evolution. Here's a simplified view of how it works:

1.  **Run & Collect:** It runs the agent with a candidate prompt on a few
    training examples to collect interaction trajectories.
2.  **Reflect:** It gives the trajectories of failed rollouts to a "reflection"
    model, which analyzes what went wrong and generates high-level insights or
    "rules" for improvement. For example, it might notice *"The agent should
    always confirm the order number before issuing a refund."*
3.  **Evolve:** It uses these insights to propose new candidate prompts by
    editing existing prompts or combining ideas from different successful ones,
    inspired by genetic algorithms.
4.  **Evaluate & Select:** It evaluates these new prompts on a validation set
    and keeps only the best-performing, diverse set of prompts (the "Pareto
    frontier").
5.  **Repeat:** It repeats this loop—collect, reflect, evolve, evaluate—until
    it reaches its budget (`max_metric_calls`).

This can result in a more detailed and robust prompt that has learned from its
mistakes, and capturing nuances that are sometimes difficult to discover
through manual prompt engineering.

## Running the experiment

The easiest way to run this demo is through the provided Colab notebook:
[`gepa_tau_bench.ipynb`](https://colab.research.google.com/github/google/adk-python/blob/main/contributing/samples/gepa/gepa_tau_bench.ipynb).

Alternatively, you can run GEPA optimization using the `run_experiment.py`
script:

```bash
python -m run_experiment \
  --output_dir=/path/to/gepa_experiments/ \
  --num_eval_trials=8 \
  --max_concurrency=32 \
  --train_batch_size=8
```

To run only evaluation with the seed prompt, use `--eval_mode`:

```bash
python -m run_experiment \
  --output_dir=/path/to/gepa_experiments/ \
  --num_eval_trials=8 \
  --max_concurrency=32 \
  --eval_mode
```

## Choosing Hyperparameters

Setting the right hyperparameters is crucial for a successful and efficient
run. The following hyperparameters can be set via command-line flags in
`run_experiment.py`:

*   `--max_metric_calls`: Total budget for GEPA prompt evaluations. This is the
    main control for runtime/cost. One could start with 100 and increase to
    500+ for further optimization.
*   `--eval_set_size`: Size of the dev set to use for Pareto frontier
    evaluation in GEPA. If None, uses all available dev tasks. A larger size
    gives a more stable, less noisy fitness score with more coverage but is
    more expensive and slows down the GEPA runtime. A few tens of examples
    might suffice for simpler tasks and up to a few hundreds
    for more complex and variable tasks.
*   `--train_batch_size`: Number of trajectories sampled from rollouts
    to be used by the reflection model in each GEPA step to generate prompt
    improvements. This corresponds to the mini-batch size in GEPA used as a
    fast, preliminary filter for new candidate prompts. It trades-off signal
    quality and cost of evaluation. The GEPA paper uses a default of 3.
    Increasing the batch size may help provide a more stable
    signal and estimate of a prompt quality but entails higher cost and less
    iterations, given a fixed budget. One can start with a low value and
    increase the size if significant variations are observed.
*   `--num_eval_trials`: Number of times each task is run during evaluation.
    Higher values give more stable evaluation metrics but increase runtime.
    Recommended: 4-8.
*   `--num_test_records`: Size of the test set for final evaluation of the
    optimized prompt. If None, uses all available test tasks.

## LLM-based Rater

When agent reward signals are not available, you can instead use an LLM rater
by setting the `--use_rater` flag.

This rater evaluates agent trajectories based on a rubric assessing whether
"The agent fulfilled the user's primary request." It provides a score (0 or 1)
and detailed feedback including evidence and rationale for its verdict. This
score is then used by GEPA as the fitness function to optimize. The rater is
implemented in `rater_lib.py`.
