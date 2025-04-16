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

try:
    import grpc as optional_grpc
except ImportError:
    optional_grpc = None

try:
    from google import auth as optional_google_auth
    from google.auth.transport import requests as optional_google_auth_requests
    from google.auth.transport import grpc as optional_google_auth_grpc
except ImportError:
    optional_google_auth = None
    optional_google_auth_requests = None
    optional_google_auth_grpc = None


def create_gcp_telemetry_api_creds():
    if optional_google_auth is None:
        return None
    if optional_grpc is None:
        return None
    creds, _ = optional_google_auth.default()
    request = optional_google_auth_requests.Request()
    auth_metadata_plugin = optional_google_auth_grpc.AuthMetadataPlugin(
        credentials=creds, request=request)
    return optional_grpc.composite_channel_credentials(
        optional_grpc.ssl_channel_credentials(),
        optional_grpc.metadata_call_credentials(auth_metadata_plugin),
    )
