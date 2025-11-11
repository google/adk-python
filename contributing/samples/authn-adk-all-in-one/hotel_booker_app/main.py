# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import wraps
import os
import sqlite3

from dotenv import load_dotenv
from flask import Flask
from flask import g
from flask import jsonify
from flask import request
from hotelbooker_core import HotelBooker
import jwt
import requests

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)
# Instantiate the core logic class
hotel_booker = HotelBooker()
app.config["DATABASE"] = hotel_booker.db_name

OIDC_CONFIG_URL = os.environ.get(
    "OIDC_CONFIG_URL", "http://localhost:5000/.well-known/openid-configuration"
)

# Cache for OIDC discovery and JWKS
oidc_config = None
jwks = None


def get_oidc_config():
  """Fetches and caches the OIDC configuration."""
  global oidc_config
  if oidc_config is None:
    try:
      response = requests.get(OIDC_CONFIG_URL)
      response.raise_for_status()
      oidc_config = response.json()
    except requests.exceptions.RequestException as e:
      return None, f"Error fetching OIDC config: {e}"
  return oidc_config, None


def get_jwks():
  """Fetches and caches the JSON Web Key Set (JWKS)."""
  global jwks
  if jwks is None:
    config, error = get_oidc_config()
    if error:
      return None, error
    jwks_uri = config.get("jwks_uri")
    if not jwks_uri:
      return None, "jwks_uri not found in OIDC configuration."
    try:
      response = requests.get(jwks_uri)
      response.raise_for_status()
      jwks = response.json()
    except requests.exceptions.RequestException as e:
      return None, f"Error fetching JWKS: {e}"
  return jwks, None


def get_db():
  """Manages a per-request database connection."""
  if "db" not in g:
    g.db = sqlite3.connect(app.config["DATABASE"])
    g.db.row_factory = sqlite3.Row
  return g.db


@app.teardown_appcontext
def close_db(exception):
  db = g.pop("db", None)
  if db is not None:
    db.close()


def is_token_valid(token: str):
  """
  Validates a JWT token using the public key from the OIDC jwks_uri.
  """
  if not token:
    return False, "Token is empty."

  jwks_data, error = get_jwks()
  if error:
    return False, f"Failed to get JWKS: {error}"

  try:
    header = jwt.get_unverified_header(token)
    kid = header.get("kid")
    if not kid:
      return False, "Token header missing 'kid'."

    key = next(
        (k for k in jwks_data.get("keys", []) if k.get("kid") == kid), None
    )
    if not key:
      return False, "No matching key found in JWKS."

    public_key = jwt.algorithms.RSAAlgorithm.from_jwk(key)

    # The decoding happens just so that we are able to
    # check if there were any exception decoding the token
    # which indicate it being not valid.
    # Also you could have verify_aud and verify_iss as False
    # But when they are true issuer and audience are needed in the jwt.decode call
    # they are checked against the values from the token
    # ideally token validation should also check whether the API being called is part of
    # audience so for example localhost:8081/api should cover localhost:8081/api/hotels
    # but should not cover localhost:8000/admin
    # so this middleware (decorator - is_token_valid, can check the request url and do that check, but we are
    # skipping that as the audience will always be localhost:8081)
    decoded_token = jwt.decode(
        token,
        key=public_key,
        issuer="http://localhost:5000",
        audience="http://localhost:8081",
        algorithms=[header["alg"]],
        options={"verify_exp": True, "verify_aud": True, "verify_iss": True},
    )
    return True, "Token is valid."
  except jwt.ExpiredSignatureError:
    return False, "Token has expired."
  except jwt.InvalidAudienceError:
    return False, "Invalid audience."
  except jwt.InvalidIssuerError:
    return False, "Invalid issuer."
  except jwt.InvalidTokenError as e:
    return False, f"Invalid token: {e}"
  except Exception as e:
    return False, f"An unexpected error occurred during token validation: {e}"


# Decorator to check for a valid access token on protected routes
def token_required(f):
  @wraps(f)
  def decorated_function(*args, **kwargs):
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
      return {
          "error": True,
          "data": None,
          "message": "Missing or invalid Authorization header.",
      }, 401

    token = auth_header.split(" ")[1]
    is_valid, message = is_token_valid(token)

    if not is_valid:
      return {"error": True, "data": None, "message": message}, 401

    return f(*args, **kwargs)

  return decorated_function


@app.route("/hotels", methods=["GET"])
@token_required
def get_hotels():
  location = request.args.get("location")
  hotels, error_message = hotel_booker.get_available_hotels(
      get_db().cursor(), location
  )

  if hotels is not None:
    return (
        jsonify({
            "error": False,
            "data": hotels,
            "message": "Successfully retrieved hotels.",
        }),
        200,
    )
  else:
    return jsonify({"error": True, "data": None, "message": error_message}), 500


@app.route("/book", methods=["POST"])
@token_required
def book_room():
  conn = get_db()
  data = request.json
  hotel_id = data.get("hotel_id")
  guest_name = data.get("guest_name")
  check_in_date = data.get("check_in_date")
  check_out_date = data.get("check_out_date")
  num_rooms = data.get("num_rooms")

  if not all([hotel_id, guest_name, check_in_date, check_out_date, num_rooms]):
    return (
        jsonify({
            "error": True,
            "data": None,
            "message": "Missing required booking information.",
        }),
        400,
    )

  booking_id, error_message = hotel_booker.book_a_room(
      conn, hotel_id, guest_name, check_in_date, check_out_date, num_rooms
  )

  if booking_id:
    return (
        jsonify({
            "error": False,
            "data": {"booking_id": booking_id},
            "message": "Booking successful!",
        }),
        200,
    )
  else:
    return jsonify({"error": True, "data": None, "message": error_message}), 400


@app.route("/booking_details", methods=["GET"])
@token_required
def get_details():
  conn = get_db()
  booking_id = request.args.get("booking_id")
  guest_name = request.args.get("guest_name")

  if not booking_id and not guest_name:
    return (
        jsonify({
            "error": True,
            "data": None,
            "message": "Please provide either a booking ID or a guest name.",
        }),
        400,
    )

  details, error_message = hotel_booker.get_booking_details(
      get_db().cursor(), booking_id=booking_id, guest_name=guest_name
  )

  if details:
    return (
        jsonify({
            "error": False,
            "data": details,
            "message": "Booking details retrieved successfully.",
        }),
        200,
    )
  else:
    return jsonify({"error": True, "data": None, "message": error_message}), 404


if __name__ == "__main__":
  app.run(debug=True, port=8081)
