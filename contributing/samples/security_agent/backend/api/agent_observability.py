"""Observability helpers for logging Chainlit sessions and running evaluations.

This module centralizes the logic for:

* Persisting user/agent exchanges into BigQuery for later analysis.
* Triggering Vertex AI GenAI Evaluations against those logged interactions.
* Recording evaluation summaries back into BigQuery for downstream reporting.

All functions assume the caller has configured the environment variables used
throughout the security agent (for example, ``GOOGLE_CLOUD_PROJECT`` and
``BQ_DEFAULT_DATASET``). Tables are created on-demand if they do not exist.
"""

from __future__ import annotations

import json
import os
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
from google.api_core import exceptions as gcloud_exceptions
from google.cloud import bigquery
from vertexai import Client
from vertexai._genai import types

# ---------------------------------------------------------------------------
# Environment configuration
# ---------------------------------------------------------------------------

PROJECT_ID = os.getenv("GOOGLE_CLOUD_PROJECT")
DATASET = os.getenv("BQ_DEFAULT_DATASET", "security_insights")
INTERACTIONS_TABLE = os.getenv(
    "AGENT_CONVERSATIONS_TABLE", "agent_conversations"
)
EVALUATIONS_TABLE = os.getenv("AGENT_EVALUATIONS_TABLE", "agent_evaluations")
LOCATION = (
    os.getenv("VERTEX_AI_LOCATION")
    or os.getenv("GOOGLE_CLOUD_LOCATION")
    or "us-central1"
)
CANDIDATE_NAME = os.getenv(
    "AGENT_EVALUATION_CANDIDATE_NAME", "security_agent_chainlit"
)

if not PROJECT_ID:
    raise RuntimeError(
        "GOOGLE_CLOUD_PROJECT must be set before importing agent_observability"
    )


# ---------------------------------------------------------------------------
# Lazy clients and helpers
# ---------------------------------------------------------------------------

_bq_client: Optional[bigquery.Client] = None
_vertex_client: Optional[Client] = None


def _get_bq_client() -> bigquery.Client:
    """Return a cached BigQuery client."""

    global _bq_client
    if _bq_client is None:
        _bq_client = bigquery.Client(project=PROJECT_ID)
    return _bq_client


def _get_vertex_client() -> Client:
    """Return a cached Vertex AI client."""

    global _vertex_client
    if _vertex_client is None:
        _vertex_client = Client(project=PROJECT_ID, location=LOCATION)
    return _vertex_client


def _full_table_name(table: str) -> str:
    return f"{PROJECT_ID}.{DATASET}.{table}"


def _ensure_interactions_table(client: bigquery.Client) -> None:
    """Create the interactions table if it does not exist."""

    table_id = _full_table_name(INTERACTIONS_TABLE)
    try:
        client.get_table(table_id)
        return
    except gcloud_exceptions.NotFound:
        pass

    schema = [
        bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("interaction_index", "INT64", mode="REQUIRED"),
        bigquery.SchemaField("user_prompt", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("agent_response", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(field="created_at")
    client.create_table(table)


def _ensure_evaluations_table(client: bigquery.Client) -> None:
    """Create the evaluations table if it does not exist."""

    table_id = _full_table_name(EVALUATIONS_TABLE)
    try:
        client.get_table(table_id)
        return
    except gcloud_exceptions.NotFound:
        pass

    schema = [
        bigquery.SchemaField("evaluation_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("session_id", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("metric_name", "STRING", mode="REQUIRED"),
        bigquery.SchemaField("mean_score", "FLOAT", mode="NULLABLE"),
        bigquery.SchemaField("num_cases_total", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("num_cases_valid", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("num_cases_error", "INT64", mode="NULLABLE"),
        bigquery.SchemaField("created_at", "TIMESTAMP", mode="REQUIRED"),
        bigquery.SchemaField("summary_json", "JSON", mode="NULLABLE"),
    ]

    table = bigquery.Table(table_id, schema=schema)
    table.time_partitioning = bigquery.TimePartitioning(field="created_at")
    client.create_table(table)


# ---------------------------------------------------------------------------
# Public logging API
# ---------------------------------------------------------------------------


def log_interaction(
    *,
    session_id: str,
    interaction_index: int,
    user_prompt: str,
    agent_response: str,
) -> None:
    """Persist a user/agent exchange into BigQuery.

    The table is created automatically if it does not exist. Errors are raised to
    the caller so they can decide whether to surface or ignore them.
    """

    client = _get_bq_client()
    _ensure_interactions_table(client)

    row = {
        "session_id": session_id,
        "interaction_index": interaction_index,
        "user_prompt": user_prompt,
        "agent_response": agent_response,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }

    table_id = _full_table_name(INTERACTIONS_TABLE)
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Failed to log interaction: {errors}")


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------


@dataclass
class _MetricSummary:
    metric_name: str
    mean_score: Optional[float]
    num_cases_total: Optional[int]
    num_cases_valid: Optional[int]
    num_cases_error: Optional[int]


_METRIC_ALIAS_MAP: Dict[str, types.Metric] = {
    "GENERAL_QUALITY": types.RubricMetric.GENERAL_QUALITY,
    "INSTRUCTION_FOLLOWING": types.RubricMetric.INSTRUCTION_FOLLOWING,
    "MULTI_TURN_CHAT_QUALITY": types.RubricMetric.MULTI_TURN_CHAT_QUALITY,
    "MULTI_TURN_GENERAL_QUALITY": types.RubricMetric.MULTI_TURN_GENERAL_QUALITY,
    "COHERENCE": types.RubricMetric.COHERENCE,
    "FLUENCY": types.RubricMetric.FLUENCY,
}


def _resolve_metric(metric: str) -> types.Metric:
    key = metric.strip().upper()
    if key in _METRIC_ALIAS_MAP:
        return _METRIC_ALIAS_MAP[key]
    raise ValueError(
        f"Unsupported metric '{metric}'. Supported metrics: {', '.join(_METRIC_ALIAS_MAP)}"
    )


def _fetch_session_rows(client: bigquery.Client, session_id: str) -> List[bigquery.Row]:
    query = f"""
        SELECT interaction_index, user_prompt, agent_response
        FROM `{PROJECT_ID}.{DATASET}.{INTERACTIONS_TABLE}`
        WHERE session_id = @session_id
        ORDER BY interaction_index
    """
    job_config = bigquery.QueryJobConfig(
        query_parameters=[bigquery.ScalarQueryParameter("session_id", "STRING", session_id)]
    )
    return list(client.query(query, job_config=job_config).result())


def _build_dataset(rows: Iterable[bigquery.Row]) -> pd.DataFrame:
    data = [
        {"prompt": row.user_prompt, "response": row.agent_response}
        for row in rows
        if row.user_prompt and row.agent_response
    ]
    return pd.DataFrame(data)


def _extract_summary(result: types.EvaluationResult) -> _MetricSummary:
    if not result.summary_metrics:
        raise ValueError("Evaluation result missing summary metrics")
    summary = result.summary_metrics[0]
    return _MetricSummary(
        metric_name=summary.metric_name,
        mean_score=summary.mean_score,
        num_cases_total=summary.num_cases_total,
        num_cases_valid=summary.num_cases_valid,
        num_cases_error=summary.num_cases_error,
    )


def _record_evaluation_result(
    client: bigquery.Client,
    session_id: str,
    summary: _MetricSummary,
    evaluation_dump: Dict[str, Any],
    evaluation_id: str,
) -> None:
    _ensure_evaluations_table(client)

    row = {
        "evaluation_id": evaluation_id,
        "session_id": session_id,
        "metric_name": summary.metric_name,
        "mean_score": summary.mean_score,
        "num_cases_total": summary.num_cases_total,
        "num_cases_valid": summary.num_cases_valid,
        "num_cases_error": summary.num_cases_error,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary_json": json.dumps(evaluation_dump),
    }

    table_id = _full_table_name(EVALUATIONS_TABLE)
    errors = client.insert_rows_json(table_id, [row])
    if errors:
        raise RuntimeError(f"Failed to log evaluation result: {errors}")


def _summarize_failures(result: types.EvaluationResult) -> List[str]:
    summaries: List[str] = []
    metric_name: Optional[str] = None
    if result.summary_metrics:
        metric_name = result.summary_metrics[0].metric_name

    for case in result.eval_case_results or []:
        response_results = getattr(case, "response_candidate_results", []) or []
        for candidate in response_results:
            metrics = getattr(candidate, "metric_results", {}) or {}
            # Prefer the primary metric if it exists
            metric_key = metric_name or next(iter(metrics), None)
            if not metric_key or metric_key not in metrics:
                continue
            metric_result = metrics[metric_key]
            score = getattr(metric_result, "score", None)
            if score is not None and score >= 1.0:
                continue
            verdicts = []
            for verdict in getattr(metric_result, "rubric_verdicts", []) or []:
                verdict_value = getattr(verdict, "verdict", "").lower()
                if verdict_value == "fail":
                    rubric = getattr(verdict, "evaluated_rubric", None)
                    rubric_title = getattr(rubric, "rubric_title", None)
                    verdicts.append(rubric_title or "Failed rubric")
            if verdicts:
                summaries.append(
                    f"Interaction {getattr(case, 'eval_case_index', '?')}: "
                    f"score={score if score is not None else 'N/A'} | Issues: {', '.join(verdicts)}"
                )
            else:
                summaries.append(
                    f"Interaction {getattr(case, 'eval_case_index', '?')}: score="
                    f"{score if score is not None else 'N/A'}"
                )
    return summaries[:5]


def run_genai_evaluation(
    *,
    session_id: str,
    metric: str = "GENERAL_QUALITY",
    candidate_name: str = CANDIDATE_NAME,
) -> str:
    """Execute a GenAI evaluation over the logged interactions for a session.

    Args:
        session_id: The Chainlit/ADK session identifier.
        metric: Human-friendly metric alias (for example, ``GENERAL_QUALITY``).
        candidate_name: Optional label that appears in Vertex evaluation reports.

    Returns:
        A human-readable summary of the evaluation outcome.
    """

    bq_client = _get_bq_client()
    rows = _fetch_session_rows(bq_client, session_id)
    if not rows:
        return (
            "⚠️ No logged interactions found for this session. "
            "Ask the user to make a few requests before running an evaluation."
        )

    df = _build_dataset(rows)
    if df.empty:
        return "⚠️ Unable to build an evaluation dataset from the logged rows."

    metric_spec = _resolve_metric(metric)
    vertex_client = _get_vertex_client()

    eval_dataset = types.EvaluationDataset(
        eval_dataset_df=df,
        candidate_name=candidate_name,
    )

    result = vertex_client.evals.evaluate(
        dataset=eval_dataset,
        metrics=[metric_spec],
    )

    summary = _extract_summary(result)
    evaluation_id = str(uuid.uuid4())
    _record_evaluation_result(
        bq_client,
        session_id=session_id,
        summary=summary,
        evaluation_dump=result.model_dump(),
        evaluation_id=evaluation_id,
    )

    score_pct = (
        f"{summary.mean_score:.1%}"
        if summary.mean_score is not None
        else "N/A"
    )
    totals = (
        f"{summary.num_cases_valid}/{summary.num_cases_total}"
        if summary.num_cases_total is not None
        else "N/A"
    )

    lines = [
        "✅ Vertex GenAI evaluation completed successfully.",
        f"• Metric: `{summary.metric_name}`",
        f"• Score: {score_pct} valid ({totals} prompts)",
        f"• Evaluation ID: `{evaluation_id}`",
        f"• Results stored in `{DATASET}.{EVALUATIONS_TABLE}`",
    ]

    failures = _summarize_failures(result)
    if failures:
        lines.append("\n⚠️ Prompts needing review:")
        lines.extend(f"  - {item}" for item in failures)

    return "\n".join(lines)


__all__ = [
    "log_interaction",
    "run_genai_evaluation",
]

