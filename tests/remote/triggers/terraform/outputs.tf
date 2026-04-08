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

output "cloud_run_url" {
  description = "Base URL of the deployed Cloud Run trigger-echo-agent service."
  value       = data.google_cloud_run_v2_service.trigger_agent.uri
}

output "pubsub_topic" {
  description = "Fully qualified Pub/Sub topic name for direct-push tests."
  value       = google_pubsub_topic.trigger_test.id
}

output "pubsub_topic_short" {
  description = "Short Pub/Sub topic name."
  value       = google_pubsub_topic.trigger_test.name
}

output "eventarc_topic" {
  description = "Fully qualified Pub/Sub topic name that fires the Eventarc trigger."
  value       = google_pubsub_topic.eventarc_source.id
}

output "eventarc_topic_short" {
  description = "Short Eventarc source topic name."
  value       = google_pubsub_topic.eventarc_source.name
}




output "project_id" {
  description = "GCP project ID used for test resources."
  value       = var.project_id
}

output "region" {
  description = "GCP region used for test resources."
  value       = var.region
}

output "suffix" {
  description = "The random suffix used for resource naming."
  value       = local.suffix
}
