"""Format query results as a markdown table."""

import sys


def parse_args(args):
  params = {}
  i = 0
  while i < len(args):
    if args[i].startswith("--") and i + 1 < len(args):
      params[args[i][2:]] = args[i + 1]
      i += 2
    elif "=" in args[i]:
      key, value = args[i].split("=", 1)
      params[key.lstrip("-")] = value
      i += 1
    else:
      i += 1
  return params


def main():
  params = parse_args(sys.argv[1:])
  header = params.get("header", "")
  rows_str = params.get("rows", "")
  title = params.get("title", "")

  if not header:
    print("Error: --header is required")
    sys.exit(1)

  cols = [c.strip() for c in header.split(",")]

  if title:
    print(f"## {title}")
    print()

  print("| " + " | ".join(cols) + " |")
  print("| " + " | ".join("---" for _ in cols) + " |")

  if rows_str:
    for row in rows_str.split(";"):
      vals = [v.strip() for v in row.split(",")]
      # Pad if needed
      while len(vals) < len(cols):
        vals.append("")
      print("| " + " | ".join(vals[: len(cols)]) + " |")


if __name__ == "__main__":
  main()
