import pytest
from google.api_core import exceptions

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

from agents._tools import service_discovery
import agents._tools.service_documentation_parser as parser_module


class _DummyBigQueryClient:
    def __init__(self, project=None):
        self.project = project

    def query(self, query):
        raise exceptions.NotFound("missing table")


def test_analyze_service_without_telemetry(monkeypatch):
    monkeypatch.setattr(service_discovery.bigquery, "Client", lambda project=None: _DummyBigQueryClient(project))
    monkeypatch.setattr(service_discovery, "HAS_ASSET_API", False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    class _StubParser:
        def parse_documentation_url(self, url, force_refresh=False):
            return {
                "description": "Managed service for serverless containers",
                "capabilities": ["security", "compute"],
                "permissions": ["run.services.get", "run.services.update"],
            }

    monkeypatch.setattr(service_discovery, "ServiceDocumentationParser", lambda *args, **kwargs: _StubParser())

    discovery = service_discovery.GCPServiceDiscovery(project_id="test-project")
    result = discovery.analyze_service("cloudrun", "security")

    assert "recommended_actions" in result
    assert any("service agent" in action for action in result["recommended_actions"])
    assert result["learned_summary"]["description"].startswith("Managed service")


def test_discover_services_includes_learned(monkeypatch, tmp_path):
    monkeypatch.setattr(service_discovery.bigquery, "Client", lambda project=None: _DummyBigQueryClient(project))
    monkeypatch.setattr(service_discovery, "HAS_ASSET_API", False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "test-project")

    class _StubParserBigQueryClient:
        def __init__(self, *args, **kwargs):
            self.project = "test"

    monkeypatch.setattr(parser_module.bigquery, "Client", _StubParserBigQueryClient)

    parser = parser_module.ServiceDocumentationParser(cache_dir=str(tmp_path))
    parser._cache_service(
        "https://example.com/service",
        {
            "service_name": "Custom Search",
            "api_endpoint": "customsearch.googleapis.com",
            "resource_types": ["indexes"],
            "capabilities": ["search"],
            "permissions": ["customsearch.indexes.get"],
            "regions": [],
        },
    )

    monkeypatch.setattr(service_discovery, "ServiceDocumentationParser", lambda *args, **kwargs: parser)

    discovery = service_discovery.GCPServiceDiscovery(project_id="test-project")
    services_report = service_discovery.discover_gcp_services(include_learned=True)

    assert "Custom Search (Learned)" in services_report
