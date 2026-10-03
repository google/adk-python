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

"""Sample demonstrating TypesafeClassifierTool with all three question types.

Triages a support ticket by asking TypeSafe AI's Jev model three independent
fixed questions -- a Choice for category, a Noul for urgency, and a Score for
severity -- rather than asking the agent's own model to reason about them in
free text.
"""

from google.adk import Agent
from google.adk.integrations.typesafe import TypesafeClassifier
from google.adk.integrations.typesafe import TypesafeClassifierTool
from typesafe_sdk import Choice
from typesafe_sdk import Noul
from typesafe_sdk import Score

classifier = TypesafeClassifier()

classify_ticket_category = TypesafeClassifierTool(
    name="classify_ticket_category",
    description=(
        "Classifies support ticket text into billing, technical, or other."
    ),
    classifier=classifier,
    question=Choice(
        instructions="What is this support ticket about?",
        criteria={
            "billing": "Charges, invoices, refunds, or payment methods.",
            "technical": "The product is not working as expected.",
            "other": "Anything that is not billing or technical.",
        },
    ),
)

check_ticket_urgency = TypesafeClassifierTool(
    name="check_ticket_urgency",
    description="Checks whether a support ticket needs a same-day response.",
    classifier=classifier,
    question=Noul(
        instructions=(
            "Does this support ticket need a same-day response, such as"
            " a service outage, a security concern, or a billing error"
            " actively costing the user money?"
        ),
        criteria={
            "true": "The user is blocked or actively losing money right now.",
            "false": "The issue can reasonably wait for a normal queue.",
        },
    ),
)

rate_ticket_severity = TypesafeClassifierTool(
    name="rate_ticket_severity",
    description="Rates how severe a support ticket's underlying issue is.",
    classifier=classifier,
    question=Score(
        instructions="How severe is the issue described in this ticket?",
        criteria=[
            (
                "Low: a minor inconvenience or a question with no functional"
                " impact."
            ),
            "Medium: a real problem, but the user has a workaround.",
            (
                "High: the user is blocked entirely or money or security is"
                " involved."
            ),
        ],
    ),
)

root_agent = Agent(
    name="support_router",
    model="gemini-2.5-flash",
    description="Triages support tickets on category, urgency, and severity.",
    instruction="""
      A user will describe a support ticket. Call classify_ticket_category,
      check_ticket_urgency, and rate_ticket_severity with the ticket text,
      then summarize all three results for the user: category, whether it is
      urgent, and its severity level with confidence.
    """,
    tools=[
        classify_ticket_category,
        check_ticket_urgency,
        rate_ticket_severity,
    ],
)
