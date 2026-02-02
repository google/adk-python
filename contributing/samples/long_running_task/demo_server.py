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

"""Custom demo server with checkpoint visualization UI."""

import asyncio
import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

from google.adk.durable import BigQueryCheckpointStore

# Configuration
PROJECT_ID = os.environ.get("GOOGLE_CLOUD_PROJECT", "test-project-0728-467323")
DATASET = "adk_metadata"
GCS_BUCKET = f"{PROJECT_ID}-adk-checkpoints"

# Initialize checkpoint store
checkpoint_store = BigQueryCheckpointStore(
    project=PROJECT_ID,
    dataset=DATASET,
    gcs_bucket=GCS_BUCKET,
)

# In-memory task state for demo
active_tasks: dict[str, dict] = {}

app = FastAPI(title="ADK Durable Session Demo")

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class TaskRequest(BaseModel):
    task_type: str  # "sentiment", "anomaly", "trend", "scan"
    duration_seconds: int = 60


class ResumeRequest(BaseModel):
    session_id: str


@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the demo UI."""
    html_path = Path(__file__).parent / "demo_ui.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Demo UI not found</h1>")


@app.get("/api/sessions")
async def list_sessions():
    """List all sessions from BigQuery."""
    try:
        client = checkpoint_store._get_bq_client()
        query = f"""
        SELECT session_id, status, agent_name, current_checkpoint_seq,
               created_at, updated_at
        FROM `{checkpoint_store._sessions_table_id}`
        ORDER BY updated_at DESC
        LIMIT 20
        """
        results = client.query(query).result()
        sessions = []
        for row in results:
            sessions.append({
                "session_id": row.session_id,
                "status": row.status,
                "agent_name": row.agent_name,
                "checkpoint_seq": row.current_checkpoint_seq,
                "created_at": row.created_at.isoformat() if row.created_at else None,
                "updated_at": row.updated_at.isoformat() if row.updated_at else None,
            })
        return {"sessions": sessions}
    except Exception as e:
        return {"sessions": [], "error": str(e)}


@app.get("/api/checkpoints/{session_id}")
async def list_checkpoints(session_id: str):
    """List checkpoints for a session."""
    try:
        checkpoints = await checkpoint_store.list_checkpoints(
            session_id=session_id, limit=20
        )
        return {
            "checkpoints": [
                {
                    "checkpoint_seq": cp.checkpoint_seq,
                    "created_at": cp.created_at.isoformat() if cp.created_at else None,
                    "trigger": cp.trigger,
                    "size_bytes": cp.size_bytes,
                    "gcs_uri": cp.gcs_state_uri,
                    "agent_state": cp.agent_state,
                }
                for cp in checkpoints
            ]
        }
    except Exception as e:
        return {"checkpoints": [], "error": str(e)}


@app.post("/api/task/start")
async def start_task(request: TaskRequest):
    """Start a new long-running task with checkpointing."""
    session_id = f"demo-{uuid.uuid4().hex[:8]}"

    # Create session in BigQuery
    try:
        session = await checkpoint_store.create_session(
            session_id=session_id,
            agent_name="demo_agent",
            metadata={"task_type": request.task_type}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create session: {e}")

    # Initialize task state
    active_tasks[session_id] = {
        "task_type": request.task_type,
        "status": "running",
        "progress": 0,
        "total_duration": request.duration_seconds,
        "records_processed": 0,
        "insights_found": 0,
        "checkpoints": [],
        "start_time": datetime.now().isoformat(),
        "should_fail": False,
        "failed_at": None,
        "final_output": None,
    }

    # Start background task
    asyncio.create_task(run_task_with_checkpoints(session_id, request.duration_seconds))

    return {
        "session_id": session_id,
        "status": "started",
        "message": f"Started {request.task_type} analysis task"
    }


@app.post("/api/task/fail/{session_id}")
async def simulate_failure(session_id: str):
    """Simulate a task failure."""
    if session_id not in active_tasks:
        raise HTTPException(status_code=404, detail="Task not found")

    active_tasks[session_id]["should_fail"] = True
    return {"status": "failure_triggered", "session_id": session_id}


@app.post("/api/task/resume")
async def resume_task(request: ResumeRequest):
    """Resume a task from checkpoint."""
    session_id = request.session_id

    # Read the latest checkpoint
    result = await checkpoint_store.read_latest_checkpoint(session_id=session_id)
    if not result:
        raise HTTPException(status_code=404, detail="No checkpoint found")

    checkpoint, state_blob = result
    state = json.loads(state_blob.decode('utf-8'))

    # Get session info
    session = await checkpoint_store.get_session(session_id=session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    # Restore task state
    active_tasks[session_id] = {
        "task_type": state.get("task_type", "unknown"),
        "status": "running",
        "progress": state.get("progress", 0),
        "total_duration": state.get("total_duration", 60),
        "records_processed": state.get("records_processed", 0),
        "insights_found": state.get("insights_found", 0),
        "checkpoints": state.get("checkpoints", []),
        "start_time": state.get("start_time"),
        "resumed_from": checkpoint.checkpoint_seq,
        "should_fail": False,
        "failed_at": None,
    }

    # Calculate remaining duration
    remaining = active_tasks[session_id]["total_duration"] * (1 - active_tasks[session_id]["progress"] / 100)

    # Resume background task
    asyncio.create_task(run_task_with_checkpoints(session_id, int(remaining), resume=True))

    return {
        "session_id": session_id,
        "status": "resumed",
        "resumed_from_checkpoint": checkpoint.checkpoint_seq,
        "progress": active_tasks[session_id]["progress"],
        "message": f"Resumed from checkpoint #{checkpoint.checkpoint_seq}"
    }


@app.get("/api/task/status/{session_id}")
async def get_task_status(session_id: str):
    """Get current task status."""
    if session_id not in active_tasks:
        # Try to get from BigQuery
        session = await checkpoint_store.get_session(session_id=session_id)
        if session:
            return {
                "session_id": session_id,
                "status": session.status,
                "checkpoint_seq": session.current_checkpoint_seq,
                "from_db": True
            }
        raise HTTPException(status_code=404, detail="Task not found")

    return {
        "session_id": session_id,
        **active_tasks[session_id]
    }


async def run_task_with_checkpoints(session_id: str, duration: int, resume: bool = False):
    """Run a long-running task with periodic checkpoints."""
    import random

    task = active_tasks.get(session_id)
    if not task:
        return

    checkpoint_interval = 10  # Checkpoint every 10 seconds
    start_progress = task["progress"] if resume else 0

    for elapsed in range(0, duration, checkpoint_interval):
        # Check if we should fail
        if task.get("should_fail"):
            task["status"] = "failed"
            task["failed_at"] = datetime.now().isoformat()
            await checkpoint_store.update_session_status(
                session_id=session_id, status="failed"
            )
            return

        # Simulate work
        await asyncio.sleep(min(checkpoint_interval, duration - elapsed))

        # Update progress
        progress = start_progress + ((elapsed + checkpoint_interval) / duration) * (100 - start_progress)
        task["progress"] = min(progress, 100)
        task["records_processed"] += random.randint(50000, 150000)
        task["insights_found"] += random.randint(1, 3)

        # Get current checkpoint seq
        session = await checkpoint_store.get_session(session_id=session_id)
        next_seq = (session.current_checkpoint_seq if session else 0) + 1

        # Create checkpoint
        state_data = {
            "task_type": task["task_type"],
            "progress": task["progress"],
            "total_duration": task["total_duration"],
            "records_processed": task["records_processed"],
            "insights_found": task["insights_found"],
            "checkpoints": task["checkpoints"],
            "start_time": task["start_time"],
        }

        try:
            checkpoint = await checkpoint_store.write_checkpoint(
                session_id=session_id,
                checkpoint_seq=next_seq,
                state_blob=json.dumps(state_data).encode('utf-8'),
                agent_state={"progress": task["progress"], "step": f"checkpoint_{next_seq}"},
                trigger="periodic",
            )

            task["checkpoints"].append({
                "seq": checkpoint.checkpoint_seq,
                "time": datetime.now().isoformat(),
                "progress": task["progress"],
            })
        except Exception as e:
            print(f"Checkpoint failed: {e}")

    # Task completed - Generate final output based on task type
    task["status"] = "completed"
    task["progress"] = 100

    # Generate realistic final output
    task_type = task.get("task_type", "analysis")
    records = task["records_processed"]
    insights = task["insights_found"]

    if task_type == "sentiment":
        task["final_output"] = {
            "title": "Sentiment Analysis Report",
            "summary": f"Analyzed {records:,} text records across multiple data sources.",
            "results": {
                "overall_sentiment": "72% Positive",
                "positive_records": int(records * 0.72),
                "neutral_records": int(records * 0.18),
                "negative_records": int(records * 0.10),
                "confidence_score": 0.94,
            },
            "key_findings": [
                "Strong positive sentiment around product quality",
                "Minor concerns about delivery times (8% of negative)",
                "Customer service mentions trending upward (+15%)",
                f"Identified {insights} actionable insights for improvement",
            ],
            "top_themes": ["quality", "value", "service", "speed", "reliability"],
            "recommendation": "Focus on delivery optimization to improve overall sentiment score by estimated 5-8%.",
        }
    elif task_type == "anomaly":
        task["final_output"] = {
            "title": "Anomaly Detection Report",
            "summary": f"Scanned {records:,} data points for unusual patterns.",
            "results": {
                "total_anomalies": insights,
                "critical_anomalies": max(1, insights // 4),
                "warning_anomalies": insights // 2,
                "info_anomalies": insights - insights // 4 - insights // 2,
                "false_positive_rate": "2.3%",
            },
            "key_findings": [
                f"Detected {insights} anomalies requiring attention",
                "3 critical anomalies in transaction processing",
                "Seasonal pattern identified in Q3 data",
                "Root cause: 67% related to system load spikes",
            ],
            "anomaly_clusters": [
                {"type": "Transaction Volume Spike", "count": 5, "severity": "high"},
                {"type": "Response Time Degradation", "count": 8, "severity": "medium"},
                {"type": "Error Rate Increase", "count": 3, "severity": "high"},
            ],
            "recommendation": "Investigate transaction processing during peak hours. Consider auto-scaling policies.",
        }
    elif task_type == "trend":
        task["final_output"] = {
            "title": "Trend Analysis Report",
            "summary": f"Analyzed {records:,} historical data points for patterns.",
            "results": {
                "trend_direction": "Upward",
                "growth_rate": "15.3% MoM",
                "seasonality_detected": True,
                "forecast_confidence": 0.89,
            },
            "key_findings": [
                "Strong upward trend detected over past 6 months",
                "15.3% month-over-month growth rate",
                "Seasonal peaks in Q4 (holiday season)",
                f"Identified {insights} significant trend changes",
            ],
            "forecast": {
                "next_month": "+12% projected",
                "next_quarter": "+38% projected",
                "confidence_interval": "±8%",
            },
            "recommendation": "Prepare for Q4 surge. Current trajectory suggests 2x capacity needed by year end.",
        }
    elif task_type == "clustering":
        task["final_output"] = {
            "title": "Data Clustering Report",
            "summary": f"Clustered {records:,} data points into meaningful segments.",
            "results": {
                "clusters_identified": 5,
                "silhouette_score": 0.78,
                "largest_cluster_size": "45%",
                "smallest_cluster_size": "8%",
            },
            "key_findings": [
                "Identified 5 distinct customer segments",
                "Largest segment (45%): 'Value Seekers'",
                "High-value segment (12%): 'Premium Customers'",
                f"Found {insights} key differentiating factors",
            ],
            "clusters": [
                {"name": "Value Seekers", "size": "45%", "description": "Price-sensitive, bulk buyers"},
                {"name": "Premium Customers", "size": "12%", "description": "High-spend, quality-focused"},
                {"name": "Occasional Shoppers", "size": "23%", "description": "Infrequent, event-driven"},
                {"name": "New Users", "size": "12%", "description": "Recent signups, exploring"},
                {"name": "Churning Risk", "size": "8%", "description": "Declining engagement"},
            ],
            "recommendation": "Target 'Churning Risk' segment with retention campaign. Estimated 15% recovery rate.",
        }
    else:
        task["final_output"] = {
            "title": "Analysis Complete",
            "summary": f"Processed {records:,} records successfully.",
            "results": {"records_processed": records, "insights_found": insights},
            "key_findings": [f"Found {insights} notable patterns in the data"],
        }

    task["final_output"]["metadata"] = {
        "session_id": session_id,
        "task_type": task_type,
        "duration_seconds": task["total_duration"],
        "checkpoints_created": len(task["checkpoints"]),
        "completed_at": datetime.now().isoformat(),
    }

    await checkpoint_store.update_session_status(
        session_id=session_id, status="completed"
    )


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
