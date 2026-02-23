"""Simulate REST API requests against a mock server."""

import json
import sys

MOCK_DB = {
    "users": [
        {"id": 1, "name": "Alice", "email": "alice@example.com"},
        {"id": 2, "name": "Bob", "email": "bob@example.com"},
        {"id": 3, "name": "Carol", "email": "carol@example.com"},
    ],
    "products": [
        {"id": 1, "name": "Laptop", "price": 999.99},
        {"id": 2, "name": "Phone", "price": 699.99},
    ],
}


def parse_args(args):
  params = {}
  for arg in args:
    if "=" in arg:
      key, value = arg.split("=", 1)
      params[key] = value
  return params


def handle_request(method, endpoint, body=None):
  method = method.upper()

  if endpoint == "/users" and method == "GET":
    return 200, MOCK_DB["users"]

  if endpoint.startswith("/users/") and method == "GET":
    user_id = int(endpoint.split("/")[-1])
    for user in MOCK_DB["users"]:
      if user["id"] == user_id:
        return 200, user
    return 404, {"error": "User not found"}

  if endpoint == "/users" and method == "POST":
    if body:
      new_user = json.loads(body)
      new_user["id"] = len(MOCK_DB["users"]) + 1
      return 201, new_user
    return 400, {"error": "Request body required"}

  if endpoint == "/products" and method == "GET":
    return 200, MOCK_DB["products"]

  return 404, {"error": f"Unknown endpoint: {method} {endpoint}"}


def main():
  params = parse_args(sys.argv[1:])
  method = params.get("method", "GET")
  endpoint = params.get("endpoint", "/users")
  body = params.get("body")

  status, response = handle_request(method, endpoint, body)
  print(json.dumps({"status": status, "data": response}, indent=2))


if __name__ == "__main__":
  main()
