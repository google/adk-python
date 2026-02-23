"""Generate Python function scaffolds from specifications."""

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
  func_name = params.get("name", "my_function")
  func_params = params.get("params", "")
  returns = params.get("returns", "None")
  description = params.get("description", "TODO: Add description")

  param_list = []
  if func_params:
    for p in func_params.split(","):
      if ":" in p:
        pname, ptype = p.split(":", 1)
        param_list.append(f"{pname}: {ptype}")
      else:
        param_list.append(p)

  params_str = ", ".join(param_list)
  print(f"def {func_name}({params_str}) -> {returns}:")
  print(f'  """{description}')
  print()
  if param_list:
    print("  Args:")
    for p in param_list:
      pname = p.split(":")[0].strip()
      print(f"    {pname}: TODO")
  print()
  print("  Returns:")
  print(f"    {returns}: TODO")
  print('  """')
  print("  raise NotImplementedError")


if __name__ == "__main__":
  main()
