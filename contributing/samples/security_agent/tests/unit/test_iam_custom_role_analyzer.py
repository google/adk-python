from types import SimpleNamespace

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

from agents._tools.iam_custom_role_analyzer import CustomRoleAnalyzer


def test_custom_role_near_match(monkeypatch):
    custom_role = SimpleNamespace(
        name="projects/test/roles/genieAdmin",
        title="Genie Admin",
        description="",
        included_permissions=[
            "discoveryengine.dataStores.get",
            "discoveryengine.dataStores.list",
            "discoveryengine.branches.list",
        ],
    )

    builtin_role = SimpleNamespace(
        name="roles/discoveryengine.admin",
        title="Discovery Engine Admin",
        description="",
        included_permissions=[
            "discoveryengine.dataStores.get",
            "discoveryengine.dataStores.list",
            "discoveryengine.branches.list",
            "discoveryengine.dataStores.create",
            "discoveryengine.servingConfigs.update",
        ],
    )

    analyzer = CustomRoleAnalyzer.__new__(CustomRoleAnalyzer)
    analyzer.project_id = "test-project"
    analyzer.dataset_id = "iam_analysis"
    analyzer.table_id = "custom_role_analysis"
    analyzer.iam_client = None
    analyzer.bq_client = None
    analyzer._builtin_role_cache = None

    monkeypatch.setattr(
        CustomRoleAnalyzer,
        "_get_custom_role",
        lambda self, role_name: custom_role,
    )
    monkeypatch.setattr(
        CustomRoleAnalyzer,
        "_get_builtin_roles",
        lambda self: [builtin_role],
    )
    monkeypatch.setattr(CustomRoleAnalyzer, "_store_analysis", lambda self, analysis: None)

    analysis = CustomRoleAnalyzer.analyze_custom_role(analyzer, "genieAdmin")

    match = analysis["best_matches"][0]
    assert match["role"] == "roles/discoveryengine.admin"
    assert match["match_type"] == "near"
    assert match["missing_count"] == 2
    assert match["missing_preview"] == [
        "discoveryengine.dataStores.create",
        "discoveryengine.servingConfigs.update",
    ]
    summary = analysis["recommendations"]["summary"]
    assert "only 2 permission" in summary
    actions = analysis["recommendations"]["actions"]
    assert any("Adopt roles/discoveryengine.admin" in action or "Adopt roles/discoveryengine.admin" in action for action in actions)

    bundle = analysis["best_role_bundle"]
    assert bundle is not None
    assert bundle["roles"][0]["role"] == "roles/discoveryengine.admin"
    assert bundle["missing_count"] == 0
    assert bundle["extra_count"] == 2
