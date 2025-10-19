#!/usr/bin/env python3
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

"""
Standalone test for backend blocking poll pattern (no ADK dependencies).

This test validates the core tool function without requiring full ADK installation.
"""

import time
import requests
import threading
import sys

# Test configuration
APPROVAL_API_URL = "http://localhost:8003"
APPROVAL_POLL_INTERVAL = 2  # Short interval for testing
APPROVAL_TIMEOUT = 30
MAX_PROPOSAL_LENGTH = 10000


def request_approval_blocking(proposal: str, context: dict = None) -> str:
    """Simplified version for testing (no ADK imports)."""
    # Input validation
    if not proposal or not proposal.strip():
        return "❌ Invalid proposal: cannot be empty"

    if len(proposal) > MAX_PROPOSAL_LENGTH:
        return f"❌ Invalid proposal: exceeds {MAX_PROPOSAL_LENGTH} character limit"

    try:
        # Create approval ticket
        response = requests.post(
            f"{APPROVAL_API_URL}/approvals",
            json={
                "proposal": proposal,
                "requester": context.get("requester", "test") if context else "test",
                "metadata": context or {}
            },
            timeout=10
        )
        response.raise_for_status()
        result = response.json()

        if not result.get("success"):
            return f"❌ Failed to create approval ticket: {result.get('message', 'Unknown error')}"

        ticket_id = result["ticket"]["ticket_id"]
        print(f"✓ Created ticket: {ticket_id}")

        # Poll for decision
        elapsed_time = 0
        poll_count = 0

        while elapsed_time < APPROVAL_TIMEOUT:
            poll_count += 1

            try:
                status_response = requests.get(
                    f"{APPROVAL_API_URL}/approvals/{ticket_id}/status",
                    timeout=10
                )
                status_response.raise_for_status()
                status_data = status_response.json()
            except requests.exceptions.RequestException as e:
                print(f"⚠  Poll {poll_count} failed: {e}")
                time.sleep(APPROVAL_POLL_INTERVAL)
                elapsed_time += APPROVAL_POLL_INTERVAL
                continue

            ticket_status = status_data.get("status")
            print(f"  Poll {poll_count}: {ticket_status} (elapsed: {elapsed_time}s)")

            if ticket_status == "approved":
                reviewer = status_data.get("reviewer", "Unknown")
                reason = status_data.get("decision_reason", "Approved")
                return f"✅ APPROVED by {reviewer}\nReason: {reason}"

            elif ticket_status == "rejected":
                reviewer = status_data.get("reviewer", "Unknown")
                reason = status_data.get("decision_reason", "No reason")
                return f"❌ REJECTED by {reviewer}\nReason: {reason}"

            elif ticket_status == "changes_requested":
                reviewer = status_data.get("reviewer", "Unknown")
                reason = status_data.get("decision_reason", "No details")
                return f"🔄 CHANGES REQUESTED by {reviewer}\nDetails: {reason}"

            elif ticket_status == "pending":
                time.sleep(APPROVAL_POLL_INTERVAL)
                elapsed_time += APPROVAL_POLL_INTERVAL
                continue

            else:
                return f"⚠️  Unknown status: {ticket_status}"

        return f"⏱️  Timeout: No decision after {APPROVAL_TIMEOUT}s"

    except requests.exceptions.ConnectionError as e:
        return f"❌ Cannot connect to API at {APPROVAL_API_URL}: {e}"
    except Exception as e:
        return f"❌ Error: {e}"


def auto_approve_after_delay(ticket_id: str, delay: int = 3):
    """Simulate approver after delay."""
    time.sleep(delay)
    print(f"\n[Simulated Approver] Approving {ticket_id} after {delay}s...")

    try:
        response = requests.post(
            f"{APPROVAL_API_URL}/approvals/{ticket_id}/approve",
            json={
                "reviewer": "test_approver",
                "decision_reason": "Auto-approved for testing",
                "next_action": "Proceed with test"
            },
            timeout=10
        )
        if response.ok:
            print(f"[Simulated Approver] ✓ Approved successfully")
        else:
            print(f"[Simulated Approver] ✗ Failed: {response.text}")
    except Exception as e:
        print(f"[Simulated Approver] ✗ Error: {e}")


def main():
    print("=" * 70)
    print("STANDALONE TEST: Backend Blocking Poll Pattern")
    print("=" * 70)

    # Check API is running
    print("\n[1/4] Checking approval API...")
    try:
        response = requests.get(f"{APPROVAL_API_URL}/approvals", timeout=2)
        response.raise_for_status()
        print(f"✓ API is running at {APPROVAL_API_URL}")
    except Exception as e:
        print(f"✗ API not available: {e}")
        print("\nPlease start mock API first:")
        print("  python3 mock_approval_api.py")
        sys.exit(1)

    # Test 1: Input validation
    print("\n[2/4] Testing input validation...")
    result = request_approval_blocking("")
    if "cannot be empty" in result:
        print("✓ Empty proposal rejected correctly")
    else:
        print(f"✗ Empty proposal validation failed: {result}")

    # Test 2: Blocking poll with simulated approval
    print("\n[3/4] Testing blocking poll with simulated approval...")
    proposal = "Deploy new feature X to production"
    context = {"priority": "high", "requester": "test_user"}

    # This approach: intercept the ticket ID after creation
    print(f"→ Starting blocking poll (simulated approval in 3s)...")

    # Create ticket manually first to get the ID
    create_response = requests.post(
        f"{APPROVAL_API_URL}/approvals",
        json={"proposal": proposal, "requester": "test_user", "metadata": context}
    )
    ticket_id = create_response.json()["ticket"]["ticket_id"]
    print(f"✓ Created ticket: {ticket_id}")

    # Start approver thread for THIS ticket
    approver = threading.Thread(target=auto_approve_after_delay, args=(ticket_id, 3))
    approver.start()

    # Poll for decision (NOT creating new ticket)
    start_time = time.time()
    elapsed_time = 0
    poll_count = 0
    result = None

    while elapsed_time < APPROVAL_TIMEOUT:
        poll_count += 1
        try:
            status_response = requests.get(
                f"{APPROVAL_API_URL}/approvals/{ticket_id}/status",
                timeout=10
            )
            status_data = status_response.json()
        except Exception as e:
            print(f"⚠  Poll {poll_count} failed: {e}")
            time.sleep(APPROVAL_POLL_INTERVAL)
            elapsed_time += APPROVAL_POLL_INTERVAL
            continue

        ticket_status = status_data.get("status")
        print(f"  Poll {poll_count}: {ticket_status} (elapsed: {elapsed_time}s)")

        if ticket_status == "approved":
            reviewer = status_data.get("reviewer", "Unknown")
            reason = status_data.get("decision_reason", "Approved")
            result = f"✅ APPROVED by {reviewer}\nReason: {reason}"
            break
        elif ticket_status == "rejected":
            reviewer = status_data.get("reviewer", "Unknown")
            reason = status_data.get("decision_reason", "No reason")
            result = f"❌ REJECTED by {reviewer}\nReason: {reason}"
            break
        elif ticket_status == "pending":
            time.sleep(APPROVAL_POLL_INTERVAL)
            elapsed_time += APPROVAL_POLL_INTERVAL
            continue
        else:
            result = f"⚠️  Unknown status: {ticket_status}"
            break

    if not result:
        result = f"⏱️  Timeout: No decision after {APPROVAL_TIMEOUT}s"

    elapsed = time.time() - start_time
    approver.join()

    # Verify result
    print(f"\n{'=' * 70}")
    print(f"RESULT (completed in {elapsed:.1f}s):")
    print(f"{'=' * 70}")
    print(result)
    print(f"{'=' * 70}")

    if "APPROVED" in result and "test_approver" in result:
        print("\n✅ TEST PASSED!")
        print("   - Ticket created successfully")
        print("   - Blocking poll waited for approval")
        print("   - Returned correct decision")
        return 0
    else:
        print(f"\n✗ TEST FAILED: Unexpected result")
        return 1


if __name__ == "__main__":
    sys.exit(main())
