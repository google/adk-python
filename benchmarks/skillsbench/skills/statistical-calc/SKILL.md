---
name: statistical-calc
description: Compute descriptive statistics (mean, median, std dev, percentiles) for numeric datasets.
---

# Statistical Calc Skill

Compute descriptive statistics on numeric data including mean, median, standard deviation, variance, and percentiles.

## Available Scripts

### `stats.py`

Computes descriptive statistics for a list of numbers.

**Usage**: `execute_skill_script(skill_name="statistical-calc", script_name="stats.py", input_args="data=10,20,30,40,50,60,70,80,90,100")`

Arguments:
- `data`: Comma-separated list of numbers

**Output format**:
```
Count: <n>
Mean: <mean>
Median: <median>
Std Dev: <std_dev>
Variance: <variance>
Min: <min>
Max: <max>
P25: <25th percentile>
P75: <75th percentile>
```

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `execute_skill_script` with numeric data to compute statistics.
3. Present the statistics to the user.
