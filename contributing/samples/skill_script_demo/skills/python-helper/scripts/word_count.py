"""Analyze word frequency in the given text."""

from collections import Counter
import sys

text = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
if not text:
  print("Error: no text provided")
  sys.exit(1)

words = text.lower().split()
freq = Counter(words)

print(f"Total words: {len(words)}")
print(f"Unique words: {len(freq)}")
print("Top words:")
for word, count in freq.most_common(5):
  print(f"  {word}: {count}")
