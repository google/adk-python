import importlib
import os
import uuid
from types import SimpleNamespace
from typing import Any, Dict
from unittest import TestCase, mock


class AgentObservabilityTests(TestCase):
    """Unit tests for backend.api.agent_observability helpers."""

    def setUp(self) -> None:
        self.env_patcher = mock.patch.dict(
            os.environ,
            {
                "GOOGLE_CLOUD_PROJECT": "test-project",
                "BQ_DEFAULT_DATASET": "test_dataset",
                "VERTEX_AI_LOCATION": "us-central1",
            },
        )
        self.env_patcher.start()

        # Import (or reload) the module under test with the patched environment.
        self.module = importlib.import_module("backend.api.agent_observability")
        importlib.reload(self.module)
        # Reset cached clients between tests.
        self.module._bq_client = None
        self.module._vertex_client = None

    def tearDown(self) -> None:
        self.env_patcher.stop()

    def test_log_interaction_inserts_row(self) -> None:
        mock_client = mock.Mock()
        mock_client.insert_rows_json.return_value = []
        with mock.patch.object(
            self.module, "_get_bq_client", return_value=mock_client
        ), mock.patch.object(self.module, "_ensure_interactions_table"):
            self.module.log_interaction(
                session_id="session-123",
                interaction_index=1,
                user_prompt="hello",
                agent_response="hi there",
            )

        mock_client.insert_rows_json.assert_called_once()
        table_id, rows = mock_client.insert_rows_json.call_args.args
        self.assertEqual(table_id, "test-project.test_dataset.agent_conversations")
        self.assertEqual(rows[0]["session_id"], "session-123")
        self.assertEqual(rows[0]["interaction_index"], 1)
        self.assertEqual(rows[0]["user_prompt"], "hello")
        self.assertEqual(rows[0]["agent_response"], "hi there")

    def test_run_genai_evaluation_records_summary(self) -> None:
        # Prepare mocked BigQuery query results
        mock_rows = [
            SimpleNamespace(
                interaction_index=1,
                user_prompt="Prompt A",
                agent_response="Response A",
            ),
            SimpleNamespace(
                interaction_index=2,
                user_prompt="Prompt B",
                agent_response="Response B",
            ),
        ]

        mock_query_job = mock.Mock()
        mock_query_job.result.return_value = mock_rows

        mock_bq_client = mock.Mock()
        mock_bq_client.insert_rows_json.return_value = []
        mock_bq_client.query.return_value = mock_query_job

        # Fake evaluation result objects
        class FakeMetricResult:
            def __init__(self, score: float) -> None:
                self.score = score
                self.rubric_verdicts = []

        class FakeCandidateResult:
            def __init__(self, score: float) -> None:
                self.metric_results = {"general_quality_v1": FakeMetricResult(score)}

        class FakeEvalCaseResult:
            def __init__(self, score: float, index: int) -> None:
                self.eval_case_index = index
                self.response_candidate_results = [FakeCandidateResult(score)]

        class FakeSummaryMetric:
            metric_name = "general_quality_v1"
            mean_score = 0.75
            num_cases_total = 2
            num_cases_valid = 2
            num_cases_error = 0

        class FakeEvaluationResult:
            def __init__(self) -> None:
                self.eval_case_results = [
                    FakeEvalCaseResult(0.75, 0),
                    FakeEvalCaseResult(1.0, 1),
                ]
                self.summary_metrics = [FakeSummaryMetric()]
                self.win_rates = None
                self.evaluation_dataset = []
                self.metadata = SimpleNamespace(candidate_names=["chainlit-agent"])

            def model_dump(self) -> Dict[str, Any]:
                return {
                    "summary_metrics": [
                        {
                            "metric_name": "general_quality_v1",
                            "mean_score": 0.75,
                        }
                    ]
                }

        mock_vertex = mock.Mock()
        mock_vertex.evals.evaluate.return_value = FakeEvaluationResult()

        with mock.patch.object(
            self.module, "_get_bq_client", return_value=mock_bq_client
        ), mock.patch.object(
            self.module, "_get_vertex_client", return_value=mock_vertex
        ), mock.patch.object(
            self.module, "_ensure_interactions_table"
        ), mock.patch.object(
            self.module, "_ensure_evaluations_table"
        ), mock.patch.object(
            self.module.uuid, "uuid4", return_value=uuid.UUID("12345678-1234-5678-1234-567812345678")
        ):
            summary = self.module.run_genai_evaluation(session_id="session-xyz")

        self.assertIn("general_quality_v1", summary)
        self.assertIn("0.75", summary)
        mock_bq_client.insert_rows_json.assert_called_with(
            "test-project.test_dataset.agent_evaluations",
            mock.ANY,
        )
