# Mock API Documentation

## Base URL

All endpoints are relative to the mock server.

## Endpoints

### GET /users
Returns a list of all users.

**Response**:
```json
[
  {"id": 1, "name": "Alice", "email": "alice@example.com"},
  {"id": 2, "name": "Bob", "email": "bob@example.com"},
  {"id": 3, "name": "Carol", "email": "carol@example.com"}
]
```

### GET /users/:id
Returns a single user by ID.

### POST /users
Creates a new user. Requires a JSON body with `name` and `email` fields.

### GET /products
Returns a list of all products.

**Response**:
```json
[
  {"id": 1, "name": "Laptop", "price": 999.99},
  {"id": 2, "name": "Phone", "price": 699.99}
]
```
