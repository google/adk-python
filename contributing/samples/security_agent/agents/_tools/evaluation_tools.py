"""Tools that integrate the Vertex AI GenAI evaluation service."""

from __future__ import annotations

from backend.api.agent_observability import run_genai_evaluation


def run_session_genai_evaluation(session_id: str, metric: str = "GENERAL_QUALITY") -> str:
    """Run a GenAI evaluation over the logged Chainlit session.

    Args:
        session_id: Full Chainlit/ADK session identifier. The welcome message in
            Chainlit displays the active session ID. Provide the exact value so
            the evaluation can retrieve the interaction log from BigQuery.
        metric: Optional evaluation metric alias (for example ``GENERAL_QUALITY``
            or ``MULTI_TURN_CHAT_QUALITY``).

    Returns:
        A human-readable summary with the evaluation score and BigQuery storage
        location for the detailed report.
    """

    return run_genai_evaluation(session_id=session_id, metric=metric)


__all__ = ["run_session_genai_evaluation"]

