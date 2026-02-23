"""Aggregate CSV data by a grouping column."""

import csv
import io
import sys


def parse_args(args):
  params = {}
  for arg in args:
    if "=" in arg:
      key, value = arg.split("=", 1)
      params[key] = value
  return params


def main():
  params = parse_args(sys.argv[1:])
  group_col = params.get("group_col", "department")
  metric_col = params.get("metric_col", "salary")

  data = """name,department,salary,years
Alice,Engineering,95000,5
Bob,Marketing,72000,3
Carol,Engineering,102000,8
Dave,Marketing,68000,2
Eve,Engineering,88000,4
Frank,Sales,76000,6
Grace,Sales,81000,7
Hank,Marketing,74000,4"""

  reader = csv.DictReader(io.StringIO(data))
  groups = {}
  for row in reader:
    group = row[group_col]
    value = float(row[metric_col])
    groups.setdefault(group, []).append(value)

  for group_name in sorted(groups):
    values = groups[group_name]
    print(f"Group: {group_name}")
    print(f"  count: {len(values)}")
    print(f"  sum: {sum(values):.0f}")
    print(f"  mean: {sum(values) / len(values):.0f}")
    print(f"  min: {min(values):.0f}")
    print(f"  max: {max(values):.0f}")


if __name__ == "__main__":
  main()
