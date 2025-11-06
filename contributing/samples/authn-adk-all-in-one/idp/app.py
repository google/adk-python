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

import base64
from datetime import datetime
from datetime import timedelta
from datetime import timezone
import hashlib
import json
import logging
import os
import time
from urllib.parse import urlencode
from urllib.parse import urlparse
from urllib.parse import urlunparse

from dotenv import load_dotenv
from flask import Flask
from flask import jsonify
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask_cors import CORS
import jwt

logging.basicConfig(level=logging.DEBUG)


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__, template_folder="templates")
CORS(app)
app.secret_key = os.urandom(24)

# Load JWKS and private key from files and environment variables
try:
  with open("jwks.json", "r") as f:
    JWKS = json.load(f)
except FileNotFoundError:
  JWKS = None
  logging.error(
      "jwks.json not found. The server will not be able to generate JWTs."
  )

PRIVATE_KEY = os.getenv("PRIVATE_KEY")
GENERATE_JWT = os.getenv("GENERATE_JWT", "true").lower() == "true"

if GENERATE_JWT and not PRIVATE_KEY:
  raise ValueError(
      "PRIVATE_KEY environment variable must be set when GENERATE_JWT is true."
  )

# A simple user registry for demonstration purposes
USER_REGISTRY = {
    "john.doe": {
        "password": "password123",
        "sub": "john.doe",
        "profile": "I am John Doe.",
        "email": "john.doe@example.com",
    },
    "jane.doe": {
        "password": "password123",
        "sub": "jane.doe",
        "profile": "I am Jane Doe.",
        "email": "jane.doe@example.com",
    },
}

OPENID_CONFIG = {
    "issuer": "http://localhost:5000",
    "authorization_endpoint": "http://localhost:5000/authorize",
    "token_endpoint": "http://localhost:5000/generate-token",
    "jwks_uri": "http://localhost:5000/jwks.json",
    "response_types_supported": ["code", "token", "id_token", "id_token token"],
    "grant_types_supported": [
        "client_credentials",
        "implicit",
        "authorization_code",
    ],
    "token_endpoint_auth_methods_supported": ["client_secret_post"],
    "scopes_supported": ["openid", "profile", "email", "api:read", "api:write"],
    "id_token_signing_alg_values_supported": ["RS256"],
    "subject_types_supported": ["public"],
    "code_challenge_methods_supported": ["S256"],
}

# A simple client registry
CLIENT_REGISTRY = {
    "abc123": {
        "client_secret": "secret123",
        "allowed_scopes": [
            "api:read",
            "api:write",
            "openid",
            "profile",
            "email",
        ],
        "redirect_uri": [
            "http://localhost:8081/callback_implicit.html",
            "http://localhost:8081/callback_authcode.html",
            "http://localhost:8081/callback_pkce.html",
            "http://localhost:8000/dev-ui/",
        ],
        "response_types": ["token", "id_token", "code"],
        "grant_types": ["client_credentials", "implicit", "authorization_code"],
        "client_name": "ADK Agent",
    }
}

# A simple "database" to store temporary authorization codes
AUTHORIZATION_CODES = {}


def generate_jwt(payload, key, alg="RS256"):
  if not JWKS:
    raise ValueError("JWKS not loaded, cannot generate JWT.")

  kid = JWKS["keys"][0]["kid"]
  headers = {"kid": kid, "alg": alg}

  return jwt.encode(payload, key, algorithm=alg, headers=headers)


def create_access_token(client_id, scopes, user_sub=None):
  if GENERATE_JWT:
    payload = {
        "iss": "http://localhost:5000",  # who issued this token?
        # aud - What client API is this token for? - please check comment in hotel booker is_token_valid
        # ideally the request's resource parameter (part of OAuth spec extension)
        # Here is an example of such request inbound to this IDP
        # GET http://localhost:5000/authorize?
        #     response_type=code&
        #     client_id=client123&
        #     redirect_uri=http%3A%2F%2Flocalhost%3A8000%2Fdev-ui&
        #     scope=openid%20profile%20api%3Aread&
        #     state=XYZ789&
        #     resource=http%3A%2F%2Flocalhost%3A8081%2Fapi
        "aud": "http://localhost:8081",
        "sub": user_sub if user_sub else client_id,
        "exp": (
            datetime.now(timezone.utc).timestamp()
            + timedelta(hours=1).total_seconds()
        ),
        "iat": datetime.now(timezone.utc).timestamp(),
        "scope": " ".join(scopes),
    }
    return generate_jwt(payload, PRIVATE_KEY)
  else:
    return os.urandom(32).hex()


def create_id_token(client_id, user_data, scopes, nonce=None):
  if not GENERATE_JWT:
    return None

  payload = {
      "iss": "http://localhost:5000",
      "sub": user_data.get("sub"),
      "aud": client_id,
      "exp": (
          datetime.now(timezone.utc).timestamp()
          + timedelta(hours=1).total_seconds()
      ),
      "iat": datetime.now(timezone.utc).timestamp(),
      "auth_time": datetime.now(timezone.utc).timestamp(),
      "email": user_data.get("email"),
      "profile": user_data.get("profile"),
      "scope": " ".join(scopes),
  }
  if nonce:
    payload["nonce"] = nonce
  return generate_jwt(payload, PRIVATE_KEY)


@app.route("/.well-known/openid-configuration")
def openid_configuration():
  return jsonify(OPENID_CONFIG)


@app.route("/jwks.json")
def jwks_endpoint():
  return jsonify(JWKS)


@app.route("/authorize", methods=["GET", "POST"])
def authorize():
  if request.method == "GET":
    client_id = request.args.get("client_id")
    redirect_uri = request.args.get("redirect_uri")
    client = CLIENT_REGISTRY.get(client_id)

    if not client or redirect_uri not in client.get("redirect_uri", []):
      return "Invalid client or redirect URI", 400

    auth_request = request.args.to_dict()
    auth_request["client_name"] = client["client_name"]
    session["auth_request"] = auth_request
    return render_template("login.html", client_name=client["client_name"])

  if request.method == "POST":
    username = request.form.get("username")
    password = request.form.get("password")
    auth_request = session.get("auth_request")
    user = USER_REGISTRY.get(username)

    if not user or user["password"] != password:
      return render_template(
          "login.html",
          error="Invalid username or password",
          client_name=auth_request["client_name"],
      )

    session["user"] = user

    return render_template("consent.html", auth_request=auth_request)


@app.route("/consent", methods=["POST"])
def consent():
  auth_request = session.get("auth_request")
  user = session.get("user")

  if not auth_request or not user:
    return "Invalid session", 400

  logging.debug(f"consent screen POST call auth_request => {auth_request}")
  client_id = auth_request.get("client_id")
  redirect_uri = auth_request.get("redirect_uri")
  scopes = auth_request.get("scope", "").split(" ")
  response_type = auth_request.get("response_type")
  state = auth_request.get("state")

  if request.form.get("consent") == "true":
    if response_type == "token id_token" or response_type == "id_token token":
      access_token = create_access_token(client_id, scopes, user.get("sub"))
      id_token = create_id_token(client_id, user, scopes)

      parsed = urlparse(redirect_uri)
      fragment_params = {
          "access_token": access_token,
          "id_token": id_token,
          "token_type": "Bearer",
          "expires_in": 3600,
          "scope": " ".join(scopes),
          "state": state,
      }
      new_uri = urlunparse((
          parsed.scheme,
          parsed.netloc,
          parsed.path,
          parsed.params,
          parsed.query,
          urlencode(fragment_params),
      ))

      session.pop("auth_request", None)
      session.pop("user", None)
      return redirect(new_uri)

    elif response_type == "code":
      auth_code = os.urandom(16).hex()
      AUTHORIZATION_CODES[auth_code] = {
          "client_id": client_id,
          "user": user,
          "scopes": scopes,
          "redirect_uri": redirect_uri,
          "expires_at": time.time() + 300,
          "code_challenge": auth_request.get("code_challenge"),
          "code_challenge_method": auth_request.get("code_challenge_method"),
      }

      parsed = urlparse(redirect_uri)
      query_params = {"code": auth_code, "state": state}
      new_uri = urlunparse((
          parsed.scheme,
          parsed.netloc,
          parsed.path,
          parsed.params,
          urlencode(query_params),
          parsed.fragment,
      ))

      session.pop("auth_request", None)
      session.pop("user", None)
      return redirect(new_uri)

  # User denied consent or invalid response
  parsed = urlparse(redirect_uri)
  query_params = {
      "error": "access_denied",
      "error_description": "User denied access",
      "state": state,
  }
  new_uri = urlunparse((
      parsed.scheme,
      parsed.netloc,
      parsed.path,
      parsed.params,
      urlencode(query_params),
      parsed.fragment,
  ))
  return redirect(new_uri)


@app.route("/generate-token", methods=["POST"])
def generate_token():
  auth_header = request.headers.get("Authorization")
  client_id = None
  client_secret = None

  # Client id and secret can come in body or in header (Authorization : Basic base64(client_id_value:client_secret_value))
  if auth_header and auth_header.startswith("Basic "):
    try:
      encoded_credentials = auth_header.split(" ")[1]
      decoded_credentials = base64.b64decode(encoded_credentials).decode(
          "utf-8"
      )
      client_id, client_secret = decoded_credentials.split(":", 1)
    except (IndexError, ValueError):
      pass  # Fallback to form data

  if not client_id or not client_secret:
    client_id = request.form.get("client_id")
    client_secret = request.form.get("client_secret")

  grant_type = request.form.get("grant_type")

  # logging.debug(f"Grant Type = {grant_type}")
  # logging.debug(f"Request => {request.__dict__}")

  client = CLIENT_REGISTRY.get(client_id)

  if not client:
    logging.error(f"invalid client {client_id}")
    return (
        jsonify(
            {"error": "invalid_client", "error_description": "Client not found"}
        ),
        401,
    )

  if client["client_secret"] != client_secret:
    logging.error(f"Client authentication failed")
    return (
        jsonify({
            "error": "invalid_client",
            "error_description": "Client authentication failed",
        }),
        401,
    )

  if grant_type == "client_credentials":
    scopes = request.form.get("scope", "").split(" ")
    for scope in scopes:
      if scope not in client["allowed_scopes"]:
        logging.error(f"Invalid_scope")
        return jsonify({"error": "invalid_scope"}), 400

    access_token = create_access_token(client_id, scopes)

    return jsonify({
        "access_token": access_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": " ".join(scopes),
    })

  elif grant_type == "authorization_code":
    code = request.form.get("code")
    redirect_uri = request.form.get("redirect_uri")
    code_verifier = request.form.get("code_verifier")

    auth_code_data = AUTHORIZATION_CODES.pop(code, None)

    if not auth_code_data:
      logging.error(f"Invalid or expired authorization code.")
      return (
          jsonify({
              "error": "invalid_grant",
              "error_description": "Invalid or expired authorization code.",
          }),
          400,
      )

    if (
        auth_code_data["redirect_uri"] != redirect_uri
        or auth_code_data["client_id"] != client_id
    ):
      logging.error(f"Redirect URI or client ID mismatch")
      return (
          jsonify({
              "error": "invalid_grant",
              "error_description": "Redirect URI or client ID mismatch",
          }),
          400,
      )

    if time.time() > auth_code_data["expires_at"]:
      logging.error(f"Authorization code has expired")
      return (
          jsonify({
              "error": "invalid_grant",
              "error_description": "Authorization code has expired",
          }),
          400,
      )

    if "code_challenge" in auth_code_data and auth_code_data["code_challenge"]:
      if not code_verifier:
        logging.error(f"Code verifier is required for PKCE flow.")
        return (
            jsonify({
                "error": "invalid_request",
                "error_description": "Code verifier is required for PKCE flow.",
            }),
            400,
        )

      computed_challenge = (
          base64.urlsafe_b64encode(
              hashlib.sha256(code_verifier.encode("utf-8")).digest()
          )
          .decode("utf-8")
          .replace("=", "")
      )
      if computed_challenge != auth_code_data["code_challenge"]:
        logging.error(f"PKCE code challenge mismatch.")
        return (
            jsonify({
                "error": "invalid_grant",
                "error_description": "PKCE code challenge mismatch.",
            }),
            400,
        )

    # Create tokens based on the stored user data
    user = auth_code_data["user"]
    access_token = create_access_token(
        client_id, auth_code_data["scopes"], user["sub"]
    )
    id_token = create_id_token(client_id, user, auth_code_data["scopes"])

    return jsonify({
        "access_token": access_token,
        "id_token": id_token,
        "token_type": "Bearer",
        "expires_in": 3600,
        "scope": " ".join(auth_code_data["scopes"]),
    })
  logging.error(f"Unsupported_grant_type")
  return jsonify({"error": "unsupported_grant_type"}), 400


@app.route("/")
def index():
  return render_template("index.html")


# --- ADMIN ROUTES START ---
@app.route("/admin")
def admin_portal():
  return render_template(
      "admin.html",
      openid_config=OPENID_CONFIG,
      user_registry=json.dumps(USER_REGISTRY),
      client_registry=json.dumps(CLIENT_REGISTRY),
  )


@app.route("/admin/update-config", methods=["POST"])
def admin_update_config():
  try:
    data = request.json
    OPENID_CONFIG["issuer"] = data.get("issuer", OPENID_CONFIG["issuer"])
    OPENID_CONFIG["authorization_endpoint"] = data.get(
        "authorization_endpoint", OPENID_CONFIG["authorization_endpoint"]
    )
    OPENID_CONFIG["jwks_uri"] = data.get("jwks_uri", OPENID_CONFIG["jwks_uri"])
    OPENID_CONFIG["token_endpoint"] = data.get(
        "token_endpoint", OPENID_CONFIG["token_endpoint"]
    )
    return jsonify(
        {"success": True, "message": "OpenID configuration updated."}
    )
  except Exception as e:
    return jsonify({"success": False, "message": str(e)}), 400


@app.route("/admin/add-user", methods=["POST"])
def admin_add_user():
  try:
    data = request.json
    username = data.get("username")
    password = data.get("password")
    sub = data.get("sub")
    profile = data.get("profile")
    email = data.get("email")

    if not username or not password or not sub:
      return (
          jsonify({
              "success": False,
              "message": "Username, password, and sub are required.",
          }),
          400,
      )

    USER_REGISTRY[username] = {
        "password": password,
        "sub": sub,
        "profile": profile,
        "email": email,
    }
    return jsonify({"success": True, "message": f"User '{username}' added."})
  except Exception as e:
    return jsonify({"success": False, "message": str(e)}), 400


@app.route("/admin/add-client", methods=["POST"])
def admin_add_client():
  try:
    data = request.json
    client_id = data.get("client_id")
    client_secret = data.get("client_secret")
    allowed_scopes = data.get("allowed_scopes", "").split()
    redirect_uri = data.get("redirect_uri", "").split()
    response_types = data.get("response_types", "").split()
    grant_types = data.get("grant_types", "").split()
    client_name = data.get("client_name")

    if not client_id or not client_name:
      return (
          jsonify({
              "success": False,
              "message": "Client ID and Client Name are required.",
          }),
          400,
      )

    CLIENT_REGISTRY[client_id] = {
        "client_secret": client_secret,
        "allowed_scopes": allowed_scopes,
        "redirect_uri": redirect_uri,
        "response_types": response_types,
        "grant_types": grant_types,
        "client_name": client_name,
    }
    return jsonify({"success": True, "message": f"Client '{client_id}' added."})
  except Exception as e:
    return jsonify({"success": False, "message": str(e)}), 400


# --- ADMIN ROUTES END ---

if __name__ == "__main__":
  app.run(port=5000)
