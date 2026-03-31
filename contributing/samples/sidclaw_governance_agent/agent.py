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

"""Sample demonstrating SidClaw governance middleware with Google ADK tools.

SidClaw adds a policy evaluation, human approval, and tamper-proof audit trail
to individual tool calls before they execute. This sample shows a customer
support agent that can send emails and access customer records. High-risk
operations — like sending an email or accessing PII — are intercepted by
SidClaw before execution. A human reviewer approves or denies each flagged
action from the SidClaw dashboard (app.sidclaw.com).

Requirements:
    pip install sidclaw google-adk

Environment variables:
    SIDCLAW_API_KEY: API key from app.sidclaw.com
    SIDCLAW_AGENT_ID: Agent ID created via `npx create-sidclaw-app` or the dashboard
"""

import os

from google.adk import Agent
from sidclaw import SidClaw
from sidclaw.middleware.google_adk import (
    GoogleADKGovernanceConfig,
    govern_google_adk_tools,
)


# ---------------------------------------------------------------------------
# Initialize SidClaw client
# ---------------------------------------------------------------------------

sidclaw_client = SidClaw(
    api_key=os.environ["SIDCLAW_API_KEY"],
    agent_id=os.environ.get("SIDCLAW_AGENT_ID", "customer-support-agent"),
)

governance_config = GoogleADKGovernanceConfig(
    data_classification={
        "send_email": "confidential",
        "get_customer_record": "confidential",
    },
    default_classification="internal",
    resource_scope="customer_support",
    wait_for_approval=True,
    approval_timeout_seconds=300.0,
)


# ---------------------------------------------------------------------------
# Tool definitions
# ---------------------------------------------------------------------------


def send_email(to: str, subject: str, body: str) -> str:
    """Send an email to a customer.

    Args:
        to: Recipient email address.
        subject: Email subject line.
        body: Plain text email body.

    Returns:
        Confirmation message with message ID.
    """
    # In production: integrate with your email provider (SendGrid, SES, etc.)
    return f"Email sent to {to} — subject: '{subject}'"


def get_customer_record(customer_id: str) -> dict:
    """Retrieve a customer record including contact information and account status.

    Args:
        customer_id: The unique customer identifier.

    Returns:
        Dictionary with customer name, email, account status, and plan.
    """
    # In production: query your CRM or database
    return {
        "customer_id": customer_id,
        "name": "Alex Johnson",
        "email": "alex@example.com",
        "status": "active",
        "plan": "business",
    }


def lookup_order(order_id: str) -> dict:
    """Look up an order by ID.

    Args:
        order_id: The order identifier.

    Returns:
        Dictionary with order details and current status.
    """
    # In production: query your order management system
    return {
        "order_id": order_id,
        "status": "shipped",
        "tracking": "1Z999AA10123456784",
        "estimated_delivery": "2026-04-05",
    }


# ---------------------------------------------------------------------------
# Wrap tools with SidClaw governance
# ---------------------------------------------------------------------------

raw_tools = [send_email, get_customer_record, lookup_order]
governed_tools = govern_google_adk_tools(
    sidclaw_client, raw_tools, governance_config
)


# ---------------------------------------------------------------------------
# ADK Agent
# ---------------------------------------------------------------------------

root_agent = Agent(
    model="gemini-2.0-flash",
    name="customer_support_agent",
    description="A customer support agent with SidClaw governance on all tools.",
    instruction="""
You are a customer support agent. Help customers with order status, account
questions, and general inquiries.

When a customer asks you to look up their order, use lookup_order.
When a customer asks about their account, use get_customer_record.
When you need to send a confirmation or follow-up email, use send_email.

Be concise and helpful. Always confirm before sending emails.
""",
    tools=governed_tools,
)
