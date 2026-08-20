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

"""Shared component-owner map for the adk_team triaging agents.

Single source of truth, imported verbatim as LABEL_TO_OWNER by BOTH
adk_triaging_agent (issues) and adk_pr_triaging_agent (PRs), so the two can
never drift. The owner becomes the issue/PR assignee (its shepherd).

github login != corp ldap, so these are the login form. Keep this in sync with
the OWNERS file, which is the authority.
"""

# Component label -> GitHub login of the owner who shepherds that component.
LABEL_TO_OWNER = {
    "agent engine": "yeesian",
    "auth": "xuanyang15",
    "bq": "shobsi",
    "cli": "wyf7107",
    "core": "DeanChensj",
    "documentation": "joefernandez",
    "eval": "i-yliu",
    "integrations": "wukath",
    "live": "wuliang229",
    "mcp": "wukath",
    "models": "xuanyang15",
    "services": "DeanChensj",
    "skills": "wukath",
    "tools": "xuanyang15",
    "tracing": "jawoszek",
    "web": "wyf7107",
    "workflow": "DeanChensj",
}
