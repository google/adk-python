# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Mock Approval API Server for Testing Backend Blocking Poll Pattern

This is a minimal FastAPI server that simulates an external approval system
for testing the backend blocking poll pattern.

## Features

- Create approval tickets
- Query ticket status
- Manual approval/rejection via API
- Simulated dashboard (HTML form)
- Auto-approval after configurable delay (for automated testing)

## Usage

```bash
# Install dependencies
pip install fastapi uvicorn

# Run server
python mock_approval_api.py

# Server starts at http://localhost:8003
# Dashboard at http://localhost:8003/
```

## API Endpoints

- POST /approvals - Create approval ticket
- GET /approvals/{ticket_id}/status - Check ticket status
- POST /approvals/{ticket_id}/approve - Approve ticket
- POST /approvals/{ticket_id}/reject - Reject ticket
- GET / - Simple dashboard for manual testing
"""

import os
import uuid
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Configuration
AUTO_APPROVE_DELAY = int(os.getenv("AUTO_APPROVE_DELAY", "0"))  # 0 = disabled
DEFAULT_PORT = int(os.getenv("APPROVAL_API_PORT", "8003"))

# Create FastAPI app
app = FastAPI(
    title="Mock Approval API",
    description="Simulated approval system for testing HITL patterns",
    version="1.0.0"
)

# In-memory ticket storage
tickets: Dict[str, Dict[str, Any]] = {}


# Request/Response models
class CreateApprovalRequest(BaseModel):
    proposal: str
    requester: str = "system"
    metadata: Optional[Dict[str, Any]] = None


class ApprovalDecisionRequest(BaseModel):
    reviewer: str
    decision_reason: str = ""
    next_action: str = ""


@app.post("/approvals")
async def create_approval(request: CreateApprovalRequest) -> Dict[str, Any]:
    """
    Create a new approval ticket.

    Returns:
        {
            "success": true,
            "ticket": {
                "ticket_id": "APR-xxxx",
                "status": "pending",
                "created_at": "ISO timestamp"
            }
        }
    """
    ticket_id = f"APR-{uuid.uuid4().hex[:8].upper()}"

    ticket = {
        "ticket_id": ticket_id,
        "proposal": request.proposal,
        "requester": request.requester,
        "metadata": request.metadata or {},
        "status": "pending",
        "created_at": datetime.now().isoformat(),
        "decision_at": None,
        "reviewer": None,
        "decision_reason": None,
        "next_action": None,
    }

    tickets[ticket_id] = ticket

    logger.info(f"Created approval ticket {ticket_id} for requester: {request.requester}")

    return {
        "success": True,
        "ticket": {
            "ticket_id": ticket_id,
            "status": ticket["status"],
            "created_at": ticket["created_at"],
        }
    }


@app.get("/approvals/{ticket_id}/status")
async def get_approval_status(ticket_id: str) -> Dict[str, Any]:
    """
    Get current status of an approval ticket.

    Returns:
        {
            "ticket_id": "APR-xxxx",
            "status": "pending" | "approved" | "rejected" | "changes_requested",
            "reviewer": "username" (if decided),
            "decision_reason": "reason text" (if decided),
            "next_action": "what to do next" (if decided)
        }
    """
    if ticket_id not in tickets:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    ticket = tickets[ticket_id]

    return {
        "ticket_id": ticket_id,
        "status": ticket["status"],
        "reviewer": ticket.get("reviewer"),
        "decision_reason": ticket.get("decision_reason"),
        "next_action": ticket.get("next_action"),
        "created_at": ticket["created_at"],
        "decision_at": ticket.get("decision_at"),
    }


@app.post("/approvals/{ticket_id}/approve")
async def approve_ticket(ticket_id: str, decision: ApprovalDecisionRequest) -> Dict[str, Any]:
    """
    Approve a ticket.

    Request Body:
        {
            "reviewer": "username",
            "decision_reason": "Looks good, tests passing",
            "next_action": "Proceed with deployment"
        }
    """
    if ticket_id not in tickets:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    ticket = tickets[ticket_id]

    if ticket["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Ticket already decided: {ticket['status']}"
        )

    ticket["status"] = "approved"
    ticket["reviewer"] = decision.reviewer
    ticket["decision_reason"] = decision.decision_reason or "Approved"
    ticket["next_action"] = decision.next_action or "Proceed"
    ticket["decision_at"] = datetime.now().isoformat()

    logger.info(
        f"Ticket {ticket_id} APPROVED by {decision.reviewer}: {decision.decision_reason}"
    )

    return {
        "success": True,
        "ticket_id": ticket_id,
        "status": "approved",
        "reviewer": ticket["reviewer"],
        "decision_reason": ticket["decision_reason"],
    }


@app.post("/approvals/{ticket_id}/reject")
async def reject_ticket(ticket_id: str, decision: ApprovalDecisionRequest) -> Dict[str, Any]:
    """
    Reject a ticket.

    Request Body:
        {
            "reviewer": "username",
            "decision_reason": "Cost too high, needs optimization",
            "next_action": "Review costs and resubmit"
        }
    """
    if ticket_id not in tickets:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    ticket = tickets[ticket_id]

    if ticket["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Ticket already decided: {ticket['status']}"
        )

    ticket["status"] = "rejected"
    ticket["reviewer"] = decision.reviewer
    ticket["decision_reason"] = decision.decision_reason or "Rejected"
    ticket["next_action"] = decision.next_action or "Review and revise"
    ticket["decision_at"] = datetime.now().isoformat()

    logger.info(
        f"Ticket {ticket_id} REJECTED by {decision.reviewer}: {decision.decision_reason}"
    )

    return {
        "success": True,
        "ticket_id": ticket_id,
        "status": "rejected",
        "reviewer": ticket["reviewer"],
        "decision_reason": ticket["decision_reason"],
    }


@app.post("/approvals/{ticket_id}/request_changes")
async def request_changes(ticket_id: str, decision: ApprovalDecisionRequest) -> Dict[str, Any]:
    """
    Request changes to a proposal.

    Request Body:
        {
            "reviewer": "username",
            "decision_reason": "Need to add error handling",
            "next_action": "Update code and resubmit"
        }
    """
    if ticket_id not in tickets:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_id} not found")

    ticket = tickets[ticket_id]

    if ticket["status"] != "pending":
        raise HTTPException(
            status_code=400,
            detail=f"Ticket already decided: {ticket['status']}"
        )

    ticket["status"] = "changes_requested"
    ticket["reviewer"] = decision.reviewer
    ticket["decision_reason"] = decision.decision_reason or "Changes requested"
    ticket["next_action"] = decision.next_action or "Make requested changes"
    ticket["decision_at"] = datetime.now().isoformat()

    logger.info(
        f"Ticket {ticket_id} CHANGES REQUESTED by {decision.reviewer}: {decision.decision_reason}"
    )

    return {
        "success": True,
        "ticket_id": ticket_id,
        "status": "changes_requested",
        "reviewer": ticket["reviewer"],
        "decision_reason": ticket["decision_reason"],
    }


@app.get("/approvals")
async def list_approvals() -> Dict[str, Any]:
    """List all approval tickets."""
    return {
        "success": True,
        "count": len(tickets),
        "tickets": [
            {
                "ticket_id": tid,
                "status": t["status"],
                "requester": t["requester"],
                "created_at": t["created_at"],
                "reviewer": t.get("reviewer"),
            }
            for tid, t in tickets.items()
        ]
    }


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """
    Simple HTML dashboard for manual approval testing.
    """
    pending_tickets = {tid: t for tid, t in tickets.items() if t["status"] == "pending"}
    all_tickets = list(tickets.items())

    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Mock Approval Dashboard</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }
            h1 { color: #333; }
            .ticket { background: white; padding: 15px; margin: 10px 0; border-radius: 5px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
            .pending { border-left: 4px solid #ff9800; }
            .approved { border-left: 4px solid #4caf50; }
            .rejected { border-left: 4px solid #f44336; }
            .changes_requested { border-left: 4px solid #2196f3; }
            .proposal { background: #f9f9f9; padding: 10px; margin: 10px 0; border-radius: 3px; font-family: monospace; white-space: pre-wrap; }
            button { padding: 8px 15px; margin: 5px; border: none; border-radius: 3px; cursor: pointer; }
            .approve { background: #4caf50; color: white; }
            .reject { background: #f44336; color: white; }
            .changes { background: #2196f3; color: white; }
            .status { display: inline-block; padding: 5px 10px; border-radius: 3px; font-weight: bold; }
            .status-pending { background: #ff9800; color: white; }
            .status-approved { background: #4caf50; color: white; }
            .status-rejected { background: #f44336; color: white; }
            .status-changes_requested { background: #2196f3; color: white; }
        </style>
    </head>
    <body>
        <h1>🎯 Mock Approval Dashboard</h1>
        <p>Total tickets: """ + str(len(all_tickets)) + """ | Pending: """ + str(len(pending_tickets)) + """</p>

        <h2>Pending Approvals</h2>
    """

    if not pending_tickets:
        html += "<p>No pending approvals.</p>"
    else:
        for tid, ticket in pending_tickets.items():
            html += f"""
            <div class="ticket pending">
                <h3>Ticket: {tid}</h3>
                <p><strong>Requester:</strong> {ticket['requester']}</p>
                <p><strong>Created:</strong> {ticket['created_at']}</p>
                <div class="proposal">{ticket['proposal']}</div>
                <form method="post" action="/approvals/{tid}/approve" style="display:inline;">
                    <button class="approve" onclick="return handleApprove('{tid}')">✅ Approve</button>
                </form>
                <form method="post" action="/approvals/{tid}/reject" style="display:inline;">
                    <button class="reject" onclick="return handleReject('{tid}')">❌ Reject</button>
                </form>
                <form method="post" action="/approvals/{tid}/request_changes" style="display:inline;">
                    <button class="changes" onclick="return handleChanges('{tid}')">🔄 Request Changes</button>
                </form>
            </div>
            """

    html += "<h2>All Tickets</h2>"

    if not all_tickets:
        html += "<p>No tickets yet.</p>"
    else:
        for tid, ticket in all_tickets:
            status_class = ticket['status'].replace('_', '-')
            html += f"""
            <div class="ticket {status_class}">
                <h3>Ticket: {tid} <span class="status status-{status_class}">{ticket['status'].upper()}</span></h3>
                <p><strong>Requester:</strong> {ticket['requester']}</p>
                <p><strong>Created:</strong> {ticket['created_at']}</p>
            """

            if ticket.get('reviewer'):
                html += f"""
                <p><strong>Reviewer:</strong> {ticket['reviewer']}</p>
                <p><strong>Decision:</strong> {ticket.get('decision_reason', 'N/A')}</p>
                <p><strong>Next Action:</strong> {ticket.get('next_action', 'N/A')}</p>
                """

            html += f"""
                <div class="proposal">{ticket['proposal']}</div>
            </div>
            """

    html += """
        <script>
            async function handleApprove(ticketId) {
                const reviewer = prompt("Your name:", "jane.smith");
                if (!reviewer) return false;
                const reason = prompt("Approval reason:", "Looks good, approved");
                const nextAction = prompt("Next action:", "Proceed");

                const response = await fetch(`/approvals/${ticketId}/approve`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({reviewer, decision_reason: reason, next_action: nextAction})
                });

                if (response.ok) {
                    alert('Approved!');
                    location.reload();
                } else {
                    alert('Error: ' + await response.text());
                }
                return false;
            }

            async function handleReject(ticketId) {
                const reviewer = prompt("Your name:", "jane.smith");
                if (!reviewer) return false;
                const reason = prompt("Rejection reason:", "Cost too high");
                const nextAction = prompt("Next action:", "Review costs and resubmit");

                const response = await fetch(`/approvals/${ticketId}/reject`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({reviewer, decision_reason: reason, next_action: nextAction})
                });

                if (response.ok) {
                    alert('Rejected!');
                    location.reload();
                } else {
                    alert('Error: ' + await response.text());
                }
                return false;
            }

            async function handleChanges(ticketId) {
                const reviewer = prompt("Your name:", "jane.smith");
                if (!reviewer) return false;
                const reason = prompt("What changes are needed?", "Add error handling");
                const nextAction = prompt("Next action:", "Update and resubmit");

                const response = await fetch(`/approvals/${ticketId}/request_changes`, {
                    method: 'POST',
                    headers: {'Content-Type': 'application/json'},
                    body: JSON.stringify({reviewer, decision_reason: reason, next_action: nextAction})
                });

                if (response.ok) {
                    alert('Changes requested!');
                    location.reload();
                } else {
                    alert('Error: ' + await response.text());
                }
                return false;
            }
        </script>
    </body>
    </html>
    """

    return html


if __name__ == "__main__":
    import uvicorn

    print(f"""
    🚀 Mock Approval API Server Starting...

    Server URL: http://localhost:{DEFAULT_PORT}
    Dashboard: http://localhost:{DEFAULT_PORT}/

    API Endpoints:
    - POST /approvals - Create approval ticket
    - GET /approvals/{{ticket_id}}/status - Check ticket status
    - POST /approvals/{{ticket_id}}/approve - Approve ticket
    - POST /approvals/{{ticket_id}}/reject - Reject ticket
    - GET /approvals - List all tickets

    Auto-approve delay: {AUTO_APPROVE_DELAY}s (0 = disabled)

    Press Ctrl+C to stop
    """)

    uvicorn.run(app, host="0.0.0.0", port=DEFAULT_PORT, log_level="info")
