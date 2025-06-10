"""
AVG Financial Expert Agent Implementation
-----------------------------------------

This package, `financial_expert_agent`, provides the concrete implementation
of a Financial Expert agent built using the Agent Versatility Gear (AVG) toolkit.

Key components include:
- `FinancialAdvisorMasterAgent`: The main orchestrator agent that understands
  user queries and routes them to appropriate expertise pipelines.
- Specialized Expert Wrappers:
  - `CompanyDataRetriever`: For fetching and structuring company-specific data.
  - `MarketNewsAnalyzer`: For analyzing market news for sentiment, summaries, etc.
  - `ReportGenerator`: For compiling data and analysis into formatted reports.

These components leverage the core AVG toolkit (`adk_avg_toolkit`) for
configuration, expert management, and pipeline execution. The custom financial
tools used by these experts are defined in the `.tools` module but are typically
accessed via expert configurations rather than direct import from this package.
"""

from .financial_expert_main_agent import FinancialAdvisorMasterAgent
from .expert_wrappers import (
    CompanyDataRetriever,
    MarketNewsAnalyzer,
    ReportGenerator
)

__all__ = [
    "FinancialAdvisorMasterAgent",
    "CompanyDataRetriever",
    "MarketNewsAnalyzer",
    "ReportGenerator",
]
