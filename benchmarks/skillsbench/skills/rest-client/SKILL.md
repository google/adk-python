---
name: rest-client
description: Simulate REST API interactions by constructing and executing HTTP-like requests.
---

# REST Client Skill

Build and execute simulated REST API requests against an embedded mock API. Demonstrates multi-step skill usage with request construction and response parsing.

## Available Scripts

### `request.py`

Executes a simulated REST API request against a mock endpoint.

**Usage**: `execute_skill_script(skill_name="rest-client", script_name="request.py", input_args="method=GET endpoint=/users")`

Supported arguments:
- `method`: HTTP method (GET, POST, PUT, DELETE)
- `endpoint`: API path (e.g., `/users`, `/users/1`, `/products`)
- `body`: JSON body for POST/PUT requests

Available mock endpoints:
- `GET /users` — List all users
- `GET /users/<id>` — Get user by ID
- `POST /users` — Create a user (requires `body`)
- `GET /products` — List all products

**Output format**: JSON response with status code

## References

- [api-docs.md](./references/api-docs.md) — Mock API documentation

## Workflow

1. Use `load_skill` to read these instructions.
2. Use `load_skill_resource` to review the API documentation.
3. Use `execute_skill_script` to make API requests.
4. Parse the response and present the data to the user.
