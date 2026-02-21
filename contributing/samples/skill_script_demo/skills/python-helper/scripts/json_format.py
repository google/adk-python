"""Pretty-print and validate a JSON string."""

import json
import sys

raw = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else ""
if not raw:
  print("Error: no JSON input provided")
  sys.exit(1)

try:
  data = json.loads(raw)
  print(json.dumps(data, indent=2, sort_keys=True))
  print(f"\nType: {type(data).__name__}")
  if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
  elif isinstance(data, list):
    print(f"Length: {len(data)}")
except json.JSONDecodeError as e:
  print(f"Invalid JSON: {e}")
  sys.exit(1)
