"""Perform regex find-and-replace on text."""

import re
import sys


def parse_args(args):
  params = {}
  current_key = None
  current_val = []
  for arg in args:
    if "=" in arg and not current_key:
      key, value = arg.split("=", 1)
      if value.startswith("'") and not value.endswith("'"):
        current_key = key
        current_val = [value[1:]]
      elif value.startswith("'") and value.endswith("'"):
        params[key] = value[1:-1]
      else:
        params[key] = value
    elif current_key:
      if arg.endswith("'"):
        current_val.append(arg[:-1])
        params[current_key] = " ".join(current_val)
        current_key = None
        current_val = []
      else:
        current_val.append(arg)
  if current_key:
    params[current_key] = " ".join(current_val)
  return params


def main():
  params = parse_args(sys.argv[1:])
  pattern = params.get("pattern", r"\d+")
  replacement = params.get("replacement", "NUM")
  text = params.get("text", "Order 123 has 45 items at $67")
  count = int(params.get("count", "0"))

  matches = re.findall(pattern, text)
  if count > 0:
    result = re.sub(pattern, replacement, text, count=count)
  else:
    result = re.sub(pattern, replacement, text)

  print(f"Original: {text}")
  print(f"Pattern: {pattern}")
  print(f"Result: {result}")
  print(f"Matches: {len(matches)}")


if __name__ == "__main__":
  main()
