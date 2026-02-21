"""Generate a Fibonacci sequence of N numbers."""

import sys

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
a, b = 0, 1
result = []
for _ in range(n):
  result.append(a)
  a, b = b, a + b
print(f"Fibonacci({n}): {result}")
print(f"Sum: {sum(result)}")
