#!/usr/bin/env python3
"""Flask web application for the BigQuery Security Agent."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import requests
from flask import (
    Flask,
    Response,
    jsonify,
    render_template,
    request,
    stream_with_context,
)

from agents.agent import root_agent
from agents._tools.base import (
    DEFAULT_DATASET,
    DEFAULT_TABLE,
    PROJECT_ID,
    StructuredToolResponse,
)
from agents._tools.security_tools import (
    get_security_insights_summary,
    get_security_statistics,
)
from agents._tools.service_discovery import (
    analyze_gcp_service,
    discover_gcp_services,
    get_service_resources,
    suggest_service_analysis,
)


ADK_BASE_URL = os.getenv("ADK_BASE_URL", "http://localhost:8000")
ADK_SESSION_URL = f"{ADK_BASE_URL}/apps/agents/users/web-user/sessions"
ADK_RUN_URL = f"{ADK_BASE_URL}/run"
STREAM_CHUNK_SIZE = int(os.getenv("STREAM_CHUNK_SIZE", "200"))
INSTRUCTION_PATH = Path(__file__).resolve().parent / "docs" / "agent_instructions.md"

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------


def create_adk_session() -> str:
    """Create a new ADK session and return its identifier."""

    try:
        response = requests.post(ADK_SESSION_URL, timeout=10)
        if response.ok:
            payload = response.json()
            return payload.get("id", str(uuid.uuid4()))
    except requests.RequestException:
        app.logger.warning("Unable to create ADK session; using random fallback", exc_info=True)
    return str(uuid.uuid4())


def extract_text_from_adk_response(result: object) -> str:
    """Extract the textual content from the ADK response payload."""

    response_text = ""
    if isinstance(result, list):
        for event in result:
            if isinstance(event, dict):
                content = event.get("content")
                if isinstance(content, dict):
                    parts = content.get("parts", [])
                    for part in parts:
                        if isinstance(part, dict) and "text" in part:
                            response_text += part["text"]
    return response_text or "No response from agent. Please try again."


def run_agent_interaction(message: str, session_id: Optional[str] = None) -> Dict[str, object]:
    """Send a message to the ADK backend and return the parsed response."""

    if not message:
        raise ValueError("No message provided")

    session_id = session_id or create_adk_session()
    payload = {
        "appName": "agents",
        "userId": "web-user",
        "sessionId": session_id,
        "newMessage": {
            "parts": [{"text": message}],
            "role": "user",
        },
    }

    response = requests.post(ADK_RUN_URL, json=payload, timeout=120)
    response.raise_for_status()

    result = response.json()
    response_text = extract_text_from_adk_response(result)
    return {"session_id": session_id, "text": response_text, "raw": result}


def chunk_text(text: str, chunk_size: int = STREAM_CHUNK_SIZE) -> Iterable[str]:
    """Yield text chunks for streaming updates."""

    if not text:
        yield ""
        return

    for index in range(0, len(text), max(1, chunk_size)):
        yield text[index : index + chunk_size]


def sse_event(data: Dict[str, object], event: Optional[str] = None) -> str:
    """Format a Server-Sent Event message."""

    payload = json.dumps(data)
    if event:
        return f"event: {event}\ndata: {payload}\n\n"
    return f"data: {payload}\n\n"


def load_instruction_markdown() -> str:
    """Load the agent instruction markdown from disk."""

    try:
        markdown = INSTRUCTION_PATH.read_text(encoding="utf-8")
        return apply_instruction_tokens(markdown)
    except FileNotFoundError:
        app.logger.warning(
            "Instruction markdown not found at %s; falling back to agent configuration",
            INSTRUCTION_PATH,
        )
        return root_agent.instruction


def apply_instruction_tokens(markdown: str) -> str:
    """Replace templated tokens in the instruction markdown."""

    replacements = {
        "{DEFAULT_DATASET}": DEFAULT_DATASET,
        "{DEFAULT_TABLE}": DEFAULT_TABLE,
        "{PROJECT_ID}": PROJECT_ID,
    }
    for token, value in replacements.items():
        markdown = markdown.replace(token, value)
    return markdown


def parse_instruction_sections(markdown_text: str) -> List[Dict[str, str]]:
    """Convert markdown headings into structured sections."""

    sections: List[Dict[str, str]] = []
    current_title = "Overview"
    current_lines: List[str] = []

    for line in markdown_text.splitlines():
        if line.startswith("# "):
            # Skip top-level heading but flush accumulated content
            if current_lines:
                sections.append(
                    {"title": current_title, "content": "\n".join(current_lines).strip()}
                )
                current_lines = []
            current_title = line[2:].strip()
            continue

        if line.startswith("## "):
            if current_lines:
                sections.append(
                    {"title": current_title, "content": "\n".join(current_lines).strip()}
                )
            current_title = line[3:].strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        sections.append({"title": current_title, "content": "\n".join(current_lines).strip()})

    return [section for section in sections if section["content"]]


def build_tool_catalog() -> List[Dict[str, str]]:
    """Return structured metadata for each registered tool."""

    catalog: List[Dict[str, str]] = []
    for tool in root_agent.tools:
        if hasattr(tool, "function"):
            func = tool.function
            catalog.append(
                {
                    "name": func.__name__,
                    "description": func.__doc__ or "No description",
                    "module": func.__module__,
                }
            )
        else:
            catalog.append(
                {"name": str(tool), "description": "Tool information not available"}
            )
    return catalog


def get_backend_status() -> str:
    """Ping the ADK backend health endpoint to determine availability."""

    try:
        response = requests.get(f"{ADK_BASE_URL}/health", timeout=5)
        return "healthy" if response.ok else "unhealthy"
    except requests.RequestException:
        return "unreachable"


def ensure_structured_payload(response: object) -> Optional[Dict[str, object]]:
    """Convert tool responses into structured dictionaries when available."""

    if isinstance(response, StructuredToolResponse):
        return response.to_dict()
    if isinstance(response, dict):
        return response
    return None


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------


@app.route("/")
def index():
    """Render the main page."""

    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    """Handle non-streaming chat requests."""

    try:
        data = request.get_json() or {}
        message = data.get("message", "")

        if not message:
            return jsonify({"error": "No message provided", "success": False}), 400

        interaction = run_agent_interaction(message)
        return jsonify({"response": interaction["text"], "success": True})
    except requests.RequestException as exc:
        app.logger.error("Error communicating with ADK backend", exc_info=True)
        return jsonify({"error": str(exc), "success": False}), 502
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.exception("Error in chat endpoint")
        return jsonify({"error": str(exc), "success": False}), 500


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """Stream chat responses via Server-Sent Events."""

    data = request.get_json() or {}
    message = data.get("message", "")

    if not message:
        return jsonify({"error": "No message provided"}), 400

    session_id = create_adk_session()

    def generate():
        yield sse_event({"session_id": session_id}, event="start")
        try:
            interaction = run_agent_interaction(message, session_id=session_id)
            text = interaction.get("text", "")
            for chunk in chunk_text(text):
                yield sse_event({"text": chunk}, event="token")
            yield sse_event({"session_id": session_id}, event="end")
        except requests.RequestException as exc:
            app.logger.error("Streaming error communicating with ADK", exc_info=True)
            yield sse_event({"message": str(exc)}, event="error")
            return
        except Exception as exc:  # pragma: no cover - defensive logging
            app.logger.exception("Unexpected error streaming chat response")
            yield sse_event({"message": str(exc)}, event="error")
            return

    response = Response(stream_with_context(generate()), mimetype="text/event-stream")
    response.headers["Cache-Control"] = "no-cache"
    response.headers["X-Accel-Buffering"] = "no"
    return response


@app.route("/health")
def health():
    """Health check endpoint."""

    try:
        backend_status = get_backend_status()
        return jsonify(
            {
                "status": "healthy",
                "agent": root_agent.name,
                "model": root_agent.model,
                "adk_backend": backend_status,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        return jsonify({"status": "unhealthy", "error": str(exc)}), 500


@app.route("/agent-info")
def agent_info():
    """Return metadata about the agent, tools, and instructions."""

    try:
        markdown = load_instruction_markdown()
        sections = parse_instruction_sections(markdown)
        preview = markdown[:200] + "..." if len(markdown) > 200 else markdown

        return jsonify(
            {
                "name": root_agent.name,
                "model": root_agent.model,
                "tools": build_tool_catalog(),
                "instruction_markdown": markdown,
                "instruction_sections": sections,
                "instruction_preview": preview,
            }
        )
    except Exception as exc:  # pragma: no cover - defensive logging
        app.logger.exception("Error building agent info payload")
        return jsonify({"error": str(exc), "success": False}), 500


@app.route("/api/metrics")
def get_metrics():
    """Return structured metrics for dashboard cards."""

    try:
        summary_response = get_security_insights_summary()
        payload = ensure_structured_payload(summary_response)

        if payload:
            data = payload.get("data", {})
            metrics = data.get("metrics", {})
            table_details = data.get("table_details", {})
            severity_stats = ensure_structured_payload(get_security_statistics("severity"))
            severity_breakdown = []
            if severity_stats:
                severity_breakdown = [
                    {
                        "severity": item.get("value"),
                        "count": item.get("count", 0),
                        "percentage": item.get("percentage", 0.0),
                    }
                    for item in severity_stats.get("data", {}).get("distribution", [])
                ]

            return jsonify(
                {
                    "total_records": metrics.get("total_records", 0),
                    "categories": metrics.get("unique_categories", 0),
                    "severity_levels": metrics.get("severity_levels", 0),
                    "resource_types": metrics.get("resource_types", 0),
                    "date_range": {
                        "earliest": metrics.get("earliest_record"),
                        "latest": metrics.get("latest_record"),
                    },
                    "table_details": table_details,
                    "severity_breakdown": severity_breakdown,
                }
            )
    except Exception as exc:
        app.logger.error("Error getting metrics", exc_info=True)

    # Fallback data for development
    return jsonify(
        {
            "total_records": 1247,
            "categories": 8,
            "severity_levels": 4,
            "resource_types": 12,
            "table_details": {"rows": 1247, "bytes": 0},
            "severity_breakdown": [
                {"severity": "CRITICAL", "count": 45, "percentage": 3.6},
                {"severity": "HIGH", "count": 234, "percentage": 18.8},
                {"severity": "MEDIUM", "count": 567, "percentage": 45.4},
                {"severity": "LOW", "count": 401, "percentage": 32.1},
            ],
        }
    )


@app.route("/api/severity-distribution")
def get_severity_distribution():
    """Return severity distribution data for charts."""

    try:
        stats_response = get_security_statistics("severity")
        payload = ensure_structured_payload(stats_response)
        if payload:
            distribution = [
                {
                    "severity": item.get("value"),
                    "count": item.get("count", 0),
                    "affected_resources": item.get("affected_resources", 0),
                    "percentage": item.get("percentage", 0.0),
                }
                for item in payload.get("data", {}).get("distribution", [])
            ]
            if distribution:
                return jsonify(distribution)
    except Exception as exc:
        app.logger.error("Error getting severity distribution", exc_info=True)

    return jsonify(
        [
            {"severity": "CRITICAL", "count": 45, "percentage": 3.6},
            {"severity": "HIGH", "count": 234, "percentage": 18.8},
            {"severity": "MEDIUM", "count": 567, "percentage": 45.4},
            {"severity": "LOW", "count": 401, "percentage": 32.1},
        ]
    )


@app.route("/api/category-distribution")
def get_category_distribution():
    """Return category distribution data for charts."""

    try:
        stats_response = get_security_statistics("category")
        payload = ensure_structured_payload(stats_response)
        if payload:
            distribution = [
                {
                    "category": item.get("value"),
                    "count": item.get("count", 0),
                    "affected_resources": item.get("affected_resources", 0),
                    "percentage": item.get("percentage", 0.0),
                }
                for item in payload.get("data", {}).get("distribution", [])
            ]
            if distribution:
                return jsonify(distribution)
    except Exception as exc:
        app.logger.error("Error getting category distribution", exc_info=True)

    return jsonify(
        [
            {"category": "IAM_POLICY", "count": 312},
            {"category": "FIREWALL_RULES", "count": 189},
            {"category": "DATA_EXPOSURE", "count": 267},
            {"category": "COMPLIANCE", "count": 145},
            {"category": "NETWORK_SECURITY", "count": 334},
        ]
    )


@app.route("/api/resource-type-distribution")
def get_resource_type_distribution():
    """Return resource type distribution data for charts."""

    try:
        stats_response = get_security_statistics("resource_type")
        payload = ensure_structured_payload(stats_response)
        if payload:
            distribution = [
                {
                    "resource_type": item.get("value"),
                    "count": item.get("count", 0),
                    "affected_resources": item.get("affected_resources", 0),
                    "percentage": item.get("percentage", 0.0),
                }
                for item in payload.get("data", {}).get("distribution", [])
            ]
            if distribution:
                return jsonify(distribution)
    except Exception as exc:
        app.logger.error("Error getting resource type distribution", exc_info=True)

    return jsonify(
        [
            {"resource_type": "compute.instances", "count": 234},
            {"resource_type": "storage.buckets", "count": 156},
            {"resource_type": "iam.serviceAccounts", "count": 289},
            {"resource_type": "container.clusters", "count": 78},
            {"resource_type": "compute.networks", "count": 123},
            {"resource_type": "bigquery.datasets", "count": 367},
        ]
    )


# ---------------------------------------------------------------------------
# Service Discovery Endpoints
# ---------------------------------------------------------------------------


@app.route("/api/services/discover", methods=["GET"])
def discover_services():
    """Discover all GCP services enabled in the project."""

    try:
        include_all = request.args.get("include_all", "false").lower() == "true"
        result = discover_gcp_services(include_all=include_all)

        if result["success"]:
            services = result.get("services", [])
            return jsonify(
                {
                    "success": True,
                    "services": services,
                    "total_count": len(services),
                    "message": f"Discovered {len(services)} services",
                }
            )
        return jsonify({"success": False, "error": result.get("error", "Discovery failed")}), 500
    except Exception as exc:
        app.logger.error("Error discovering services", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/services/analyze", methods=["POST"])
def analyze_service():
    """Perform on-demand analysis of a specific GCP service."""

    try:
        data = request.get_json() or {}
        service_name = data.get("service_name", "")
        analysis_types = data.get("analysis_types", ["security", "compliance"])
        custom_query = data.get("custom_query")

        if not service_name:
            return jsonify({"success": False, "error": "Service name is required"}), 400

        analysis_query = json.dumps(
            {"service": service_name, "types": analysis_types, "custom_query": custom_query}
        )

        result = analyze_gcp_service(service_name=service_name, analysis_query=analysis_query)

        if result["success"]:
            analysis_data = result.get("analysis", {})
            findings: List[Dict[str, object]] = []

            if "security" in analysis_types:
                for finding in analysis_data.get("security_findings", [])[:5]:
                    findings.append(
                        {
                            "type": "security",
                            "severity": finding.get("severity", "INFO"),
                            "title": finding.get("title", "Security Finding"),
                            "description": finding.get("description", ""),
                            "recommendation": finding.get("recommendation", ""),
                        }
                    )

            return jsonify(
                {
                    "success": True,
                    "service": service_name,
                    "analysis": analysis_data,
                    "findings": findings,
                    "message": f"Analysis complete for {service_name}",
                }
            )

        return jsonify({"success": False, "error": result.get("error", "Analysis failed")}), 500
    except Exception as exc:
        app.logger.error("Error analyzing service", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/services/resources/<service_name>", methods=["GET"])
def get_resources(service_name):
    """Get resources for a specific service."""

    try:
        resource_type = request.args.get("resource_type")
        limit = int(request.args.get("limit", 100))

        result = get_service_resources(
            service_name=service_name,
            resource_type=resource_type,
            limit=limit,
        )

        if result["success"]:
            resources = result.get("resources", [])
            return jsonify(
                {
                    "success": True,
                    "service": service_name,
                    "resources": resources,
                    "count": len(resources),
                }
            )
        return jsonify({"success": False, "error": result.get("error", "Failed to get resources")}), 500
    except Exception as exc:
        app.logger.error("Error getting resources", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/services/suggest", methods=["GET"])
def suggest_analysis():
    """Get AI-powered suggestions for service analysis."""

    try:
        query = request.args.get("query", "")

        if not query:
            return jsonify({"success": False, "error": "Query is required"}), 400

        result = suggest_service_analysis(user_query=query)

        if result["success"]:
            suggestions = result.get("suggestions", [])
            recommendations = []
            for index, suggestion in enumerate(suggestions[:5], 1):
                recommendations.append(
                    {
                        "id": index,
                        "title": suggestion.get("title", f"Analysis {index}"),
                        "description": suggestion.get("description", ""),
                        "query": suggestion.get("query", ""),
                        "service": suggestion.get("service", ""),
                        "priority": suggestion.get("priority", "Medium"),
                        "estimated_time": suggestion.get("estimated_time", "< 1 minute"),
                    }
                )

            return jsonify({"success": True, "query": query, "recommendations": recommendations})

        return jsonify({"success": False, "error": result.get("error", "Failed to get suggestions")}), 500
    except Exception as exc:
        app.logger.error("Error getting suggestions", exc_info=True)
        return jsonify({"success": False, "error": str(exc)}), 500


@app.route("/api/services/categories", methods=["GET"])
def get_service_categories():
    """Get available service categories for filtering."""

    categories = [
        {"id": "compute", "name": "Compute", "icon": "💻", "count": 0},
        {"id": "storage", "name": "Storage", "icon": "💾", "count": 0},
        {"id": "database", "name": "Database", "icon": "🗄️", "count": 0},
        {"id": "networking", "name": "Networking", "icon": "🌐", "count": 0},
        {"id": "ai-ml", "name": "AI & ML", "icon": "🤖", "count": 0},
        {"id": "analytics", "name": "Analytics", "icon": "📊", "count": 0},
        {"id": "security", "name": "Security", "icon": "🔒", "count": 0},
        {"id": "management", "name": "Management", "icon": "⚙️", "count": 0},
        {"id": "developer", "name": "Developer Tools", "icon": "🛠️", "count": 0},
        {"id": "integration", "name": "Integration", "icon": "🔗", "count": 0},
    ]

    return jsonify({"success": True, "categories": categories})


if __name__ == "__main__":
    print("🚀 Starting Flask app for BigQuery Security Agent")
    print(f"   Agent: {root_agent.name}")
    print(f"   Model: {root_agent.model}")
    print(f"   Tools: {len(root_agent.tools)} tools available")
    print("\n📍 Server running at: http://localhost:5000")
    print("   Health check: http://localhost:5000/health")
    print("   Agent info: http://localhost:5000/agent-info")

    app.run(debug=True, port=5000)
