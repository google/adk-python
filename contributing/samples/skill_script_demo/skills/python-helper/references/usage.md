# Python Helper Usage Guide

## Quick Reference

| Script | Purpose | Example Args |
|--------|---------|-------------|
| `fibonacci.py` | Generate Fibonacci sequence | `"15"` (count) |
| `word_count.py` | Word frequency analysis | `"hello world hello"` |
| `json_format.py` | Validate & pretty-print JSON | `'{"key":"value"}'` |

## Tips

- All scripts write results to **stdout**.
- Pass arguments via the `input_args` parameter as a space-separated string.
- `fibonacci.py` defaults to 10 numbers if no argument is given.
- `word_count.py` treats all arguments as the text to analyze.
- `json_format.py` joins all arguments as a single JSON string.
