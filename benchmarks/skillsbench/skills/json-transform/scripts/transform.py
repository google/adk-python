"""Transform JSON by flattening nested structures."""

import json
import sys

SAMPLE_DATA = {
    "user": {
        "name": "Alice",
        "age": 30,
        "address": {
            "city": "Seattle",
            "state": "WA",
            "zip": "98101",
        },
    },
    "orders": [
        {"id": 1, "total": 42.50},
        {"id": 2, "total": 18.75},
    ],
}


def flatten(obj, prefix=""):
  result = {}
  for key, value in obj.items():
    full_key = f"{prefix}.{key}" if prefix else key
    if isinstance(value, dict):
      result.update(flatten(value, full_key))
    else:
      result[full_key] = value
  return result


def parse_args(args):
  params = {}
  for arg in args:
    if "=" in arg:
      key, value = arg.split("=", 1)
      params[key] = value
  return params


def main():
  params = parse_args(sys.argv[1:])
  data = SAMPLE_DATA.copy()

  if params.get("flatten", "").lower() == "true":
    data = flatten(data)

  if "keep" in params:
    keep_fields = set(params["keep"].split(","))
    data = {k: v for k, v in data.items() if k in keep_fields}

  if "rename" in params:
    old_name, new_name = params["rename"].split(":", 1)
    if old_name in data:
      data[new_name] = data.pop(old_name)

  print(json.dumps(data, indent=2))


if __name__ == "__main__":
  main()
