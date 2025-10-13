import pytest

import sys
import types
from pathlib import Path

_agents_path = Path(__file__).resolve().parents[2] / "agents"
if "agents" not in sys.modules:
    _agents_pkg = types.ModuleType("agents")
    _agents_pkg.__path__ = [str(_agents_path)]
    sys.modules["agents"] = _agents_pkg

if "unified_data_api" not in sys.modules:
    uda_module = types.ModuleType("unified_data_api")

    class _DummyBigQueryOperations:
        def __init__(self, *args, **kwargs):
            pass

        def query_to_models(self, query, model_cls):
            return []

    uda_module.BigQueryOperations = _DummyBigQueryOperations

    models_module = types.ModuleType("unified_data_api.models")
    for name in [
        "IAMAccount",
        "FirewallRule",
        "StorageBucket",
        "SecurityFinding",
        "Severity",
        "AccountType",
    ]:
        setattr(models_module, name, type(name, (), {}))

    uda_module.models = models_module
    sys.modules["unified_data_api"] = uda_module
    sys.modules["unified_data_api.models"] = models_module

from agents._tools.service_documentation_parser import ServiceDocumentationParser
import agents._tools.service_documentation_parser as parser_module


class _DummyResponse:
    def __init__(self, text: str):
        self.text = text

    def raise_for_status(self):
        return None


def test_parse_documentation_filters_noise(monkeypatch, tmp_path):
    html = """
    <html>
      <head><title>Cloud Run | Google Cloud</title></head>
      <body>
        <h1>Cloud Run</h1>
        <p>Capabilities include security, compute, storage.</p>
        <p>Required permission: discoveryengine.dataStores.create</p>
        <p>Visit https://www.youtube.com/watch?v=123 for a demo.</p>
      </body>
    </html>
    """

    monkeypatch.setattr(
        parser_module.requests,
        "get",
        lambda url, timeout=30: _DummyResponse(html),
    )

    # Force BigQuery client creation to fail so we operate offline during tests
    class _StubBigQueryClient:
        def __init__(self, *args, **kwargs):
            self.project = "test"

    monkeypatch.setattr(parser_module.bigquery, "Client", _StubBigQueryClient)

    monkeypatch.setattr(ServiceDocumentationParser, "_store_in_bigquery", lambda self, info: None)
    parser = ServiceDocumentationParser(cache_dir=str(tmp_path))
    result = parser.parse_documentation_url("https://cloud.google.com/run/docs")

    permissions = result.get("permissions", [])
    assert "discoveryengine.dataStores.create" in permissions
    assert all("youtube" not in perm for perm in permissions)
