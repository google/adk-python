"""Compute descriptive statistics for numeric data."""

import math
import sys


def parse_args(args):
  params = {}
  for arg in args:
    if "=" in arg:
      key, value = arg.split("=", 1)
      params[key] = value
  return params


def percentile(sorted_data, p):
  k = (len(sorted_data) - 1) * (p / 100.0)
  f = math.floor(k)
  c = math.ceil(k)
  if f == c:
    return sorted_data[int(k)]
  return sorted_data[int(f)] * (c - k) + sorted_data[int(c)] * (k - f)


def main():
  params = parse_args(sys.argv[1:])
  data_str = params.get("data", "10,20,30,40,50,60,70,80,90,100")
  data = [float(x.strip()) for x in data_str.split(",")]

  n = len(data)
  mean = sum(data) / n
  sorted_data = sorted(data)

  if n % 2 == 0:
    median = (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
  else:
    median = sorted_data[n // 2]

  variance = sum((x - mean) ** 2 for x in data) / n
  std_dev = math.sqrt(variance)

  print(f"Count: {n}")
  print(f"Mean: {mean:.2f}")
  print(f"Median: {median:.2f}")
  print(f"Std Dev: {std_dev:.2f}")
  print(f"Variance: {variance:.2f}")
  print(f"Min: {min(data):.2f}")
  print(f"Max: {max(data):.2f}")
  print(f"P25: {percentile(sorted_data, 25):.2f}")
  print(f"P75: {percentile(sorted_data, 75):.2f}")


if __name__ == "__main__":
  main()
