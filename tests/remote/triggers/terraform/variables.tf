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

variable "project_id" {
  description = "GCP project ID for test resources."
  type        = string
}

variable "region" {
  description = "GCP region for Cloud Run and related resources."
  type        = string
  default     = "us-central1"
}

variable "simulate_429_count" {
  description = "Number of 429 errors the echo agent simulates before succeeding (0 = disabled)."
  type        = number
  default     = 0
}

variable "service_name" {
  description = "Optional name of a pre-deployed Cloud Run service. If provided, Terraform will use it instead of generating a name from the suffix."
  type        = string
  default     = null
}

variable "suffix" {
  description = "Optional suffix for resource naming. If not provided, a random one will be generated."
  type        = string
  default     = null
}
