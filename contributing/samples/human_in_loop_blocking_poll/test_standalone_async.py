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
Standalone async test for backend blocking poll pattern (no ADK dependencies).

This test validates the async version without requiring full ADK installation.
"""

import asyncio
import aiohttp
import time
import sys

# Test configuration
APPROVAL_API_URL = "http://localhost:8003"
APPROVAL_POLL_INTERVAL = 2  # Short interval for testing
APPROVAL_TIMEOUT = 30
MAX_PROPOSAL_LENGTH = 10000


async def request_approval_blocking_async(proposal: str, context: dict = None) -> str:
    """Async version for testing (no ADK imports)."""
    # Input validation
    if not proposal or not proposal.strip():
        return "❌ Invalid proposal: cannot be empty"

    if len(proposal) > MAX_PROPOSAL_LENGTH:
        return f"❌ Invalid proposal: exceeds {MAX_PROPOSAL_LENGTH} character limit"

    try:
        async with aiohttp.ClientSession() as session:
            # Create approval ticket
            async with session.post(
                f"{APPROVAL_API_URL}/approvals",
                json={
                    "proposal": proposal,
                    "requester": context.get("requester", "test") if context else "test",
                    "metadata": context or {}
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                response.raise_for_status()
                result = await response.json()

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
                    async with session.get(
                        f"{APPROVAL_API_URL}/approvals/{ticket_id}/status",
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as status_response:
                        status_response.raise_for_status()
                        status_data = await status_response.json()
                except aiohttp.ClientError as e:
                    print(f"⚠  Poll {poll_count} failed: {e}")
                    await asyncio.sleep(APPROVAL_POLL_INTERVAL)
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
                    await asyncio.sleep(APPROVAL_POLL_INTERVAL)
                    elapsed_time += APPROVAL_POLL_INTERVAL
                    continue

                else:
                    return f"⚠️  Unknown status: {ticket_status}"

            return f"⏱️  Timeout: No decision after {APPROVAL_TIMEOUT}s"

    except aiohttp.ClientConnectorError as e:
        return f"❌ Cannot connect to API at {APPROVAL_API_URL}: {e}"
    except Exception as e:
        return f"❌ Error: {e}"


async def auto_approve_after_delay(ticket_id: str, delay: int = 3):
    """Simulate approver after delay."""
    await asyncio.sleep(delay)
    print(f"\n[Simulated Approver] Approving {ticket_id} after {delay}s...")

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{APPROVAL_API_URL}/approvals/{ticket_id}/approve",
                json={
                    "reviewer": "test_approver_async",
                    "decision_reason": "Auto-approved for async testing",
                    "next_action": "Proceed with async test"
                },
                timeout=aiohttp.ClientTimeout(total=10)
            ) as response:
                if response.ok:
                    print(f"[Simulated Approver] ✓ Approved successfully")
                else:
                    text = await response.text()
                    print(f"[Simulated Approver] ✗ Failed: {text}")
    except Exception as e:
        print(f"[Simulated Approver] ✗ Error: {e}")


async def main():
    print("=" * 70)
    print("STANDALONE ASYNC TEST: Backend Blocking Poll Pattern")
    print("=" * 70)

    # Check API is running
    print("\n[1/4] Checking approval API...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(
                f"{APPROVAL_API_URL}/approvals",
                timeout=aiohttp.ClientTimeout(total=2)
            ) as response:
                response.raise_for_status()
        print(f"✓ API is running at {APPROVAL_API_URL}")
    except Exception as e:
        print(f"✗ API not available: {e}")
        print("\nPlease start mock API first:")
        print("  python3 mock_approval_api.py")
        sys.exit(1)

    # Test 1: Input validation
    print("\n[2/4] Testing input validation...")
    result = await request_approval_blocking_async("")
    if "cannot be empty" in result:
        print("✓ Empty proposal rejected correctly")
    else:
        print(f"✗ Empty proposal validation failed: {result}")

    # Test 2: Blocking poll with simulated approval
    print("\n[3/4] Testing async blocking poll with simulated approval...")
    proposal = "Deploy new feature X to production (async test)"
    context = {"priority": "high", "requester": "test_user_async"}

    print(f"→ Starting async blocking poll (simulated approval in 3s)...")

    # Create ticket manually first to get the ID
    async with aiohttp.ClientSession() as session:
        async with session.post(
            f"{APPROVAL_API_URL}/approvals",
            json={"proposal": proposal, "requester": "test_user_async", "metadata": context}
        ) as create_response:
            create_result = await create_response.json()
            ticket_id = create_result["ticket"]["ticket_id"]
    print(f"✓ Created ticket: {ticket_id}")

    # Start approver task for THIS ticket
    approver_task = asyncio.create_task(auto_approve_after_delay(ticket_id, 3))

    # Poll for decision (NOT creating new ticket)
    start_time = time.time()
    elapsed_time = 0
    poll_count = 0
    result = None

    async with aiohttp.ClientSession() as session:
        while elapsed_time < APPROVAL_TIMEOUT:
            poll_count += 1
            try:
                async with session.get(
                    f"{APPROVAL_API_URL}/approvals/{ticket_id}/status",
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as status_response:
                    status_data = await status_response.json()
            except Exception as e:
                print(f"⚠  Poll {poll_count} failed: {e}")
                await asyncio.sleep(APPROVAL_POLL_INTERVAL)
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
                await asyncio.sleep(APPROVAL_POLL_INTERVAL)
                elapsed_time += APPROVAL_POLL_INTERVAL
                continue
            else:
                result = f"⚠️  Unknown status: {ticket_status}"
                break

    if not result:
        result = f"⏱️  Timeout: No decision after {APPROVAL_TIMEOUT}s"

    elapsed = time.time() - start_time
    await approver_task

    # Verify result
    print(f"\n{'=' * 70}")
    print(f"RESULT (completed in {elapsed:.1f}s):")
    print(f"{'=' * 70}")
    print(result)
    print(f"{'=' * 70}")

    if "APPROVED" in result and "test_approver_async" in result:
        print("\n✅ TEST PASSED!")
        print("   - Ticket created successfully")
        print("   - Async blocking poll waited for approval (non-blocking)")
        print("   - Returned correct decision")
        return 0
    else:
        print(f"\n✗ TEST FAILED: Unexpected result")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
