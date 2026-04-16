# Copyright 2026 Google LLC
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

# ---------------------------------------------------------------------------
# Pub/Sub topic + push subscription pointing to /trigger/pubsub
# ---------------------------------------------------------------------------

resource "google_pubsub_topic" "trigger_test" {
  name    = "${local.name_prefix}-${local.suffix}"
  project = var.project_id

  depends_on = [google_project_service.apis]
}

resource "google_pubsub_subscription" "trigger_push" {
  name    = "${local.name_prefix}-push-${local.suffix}"
  project = var.project_id
  topic   = google_pubsub_topic.trigger_test.id

  push_config {
    push_endpoint = "${data.google_cloud_run_v2_service.trigger_agent.uri}/apps/trigger_echo_agent/trigger/pubsub"

    oidc_token {
      service_account_email = google_service_account.pubsub_invoker.email
      audience              = data.google_cloud_run_v2_service.trigger_agent.uri
    }
  }

  # Short ack deadline for faster test feedback.
  ack_deadline_seconds = 30

  # Retry policy for failed pushes.
  retry_policy {
    minimum_backoff = "10s"
    maximum_backoff = "60s"
  }

  # Let the subscription persist until Terraform destroys it.
  # (default retention of 604800s is fine for tests)
  expiration_policy {
    ttl = ""
  }
}
