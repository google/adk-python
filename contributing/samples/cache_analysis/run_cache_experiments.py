#!/usr/bin/env python3
# Copyright 2025 Google LLC
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

"""
Cache Performance Experiments for ADK Context Caching

This script runs two experiments to compare caching performance:
A. Gemini 2.0 Flash: Cache enabled vs disabled (explicit caching test)
B. Gemini 2.5 Flash: Implicit vs explicit caching comparison
"""

import argparse
import asyncio
import copy
import json
import logging
import sys
import time
from typing import Any
from typing import Dict
from typing import List

try:
  # Try relative imports first (when run as module)
  from .agent import app
  from .utils import get_test_prompts
  from .utils import run_experiment_batch
except ImportError:
  # Fallback to direct imports (when run as script)
  from agent import app
  from utils import get_test_prompts
  from utils import run_experiment_batch

from google.adk.cli.utils import logs
from google.adk.runners import InMemoryRunner
from google.adk.utils.cache_performance_analyzer import CachePerformanceAnalyzer

APP_NAME = "cache_analysis_experiments"
USER_ID = "cache_researcher"


def create_agent_variant(base_app, model_name: str, cache_enabled: bool):
  """Create an app variant with specified model and cache settings."""
  import datetime

  from google.adk.agents.context_cache_config import ContextCacheConfig
  from google.adk.apps.app import App

  # Extract the root agent and modify its model
  agent_copy = copy.deepcopy(base_app.root_agent)
  agent_copy.model = model_name

  # Prepend dynamic timestamp to instruction to avoid implicit cache reuse across runs
  current_timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
  dynamic_prefix = f"Current session started at: {current_timestamp}\n\n"
  agent_copy.instruction = dynamic_prefix + agent_copy.instruction

  # Update agent name to reflect configuration
  cache_status = "cached" if cache_enabled else "no_cache"
  agent_copy.name = (
      f"cache_analysis_{model_name.replace('.', '_').replace('-', '_')}_{cache_status}"
  )

  if cache_enabled:
    # Use standardized cache config
    cache_config = ContextCacheConfig(
        min_tokens=4096,
        ttl_seconds=600,  # 10 mins for research sessions
        cache_intervals=3,  # Maximum invocations before cache refresh
    )
  else:
    # Disable caching by setting config to None
    cache_config = None

  # Create new App with updated configuration
  app_copy = App(
      name=f"{base_app.name}_{cache_status}",
      root_agent=agent_copy,
      context_cache_config=cache_config,
  )

  return app_copy


async def run_cache_comparison_experiment(
    model_name: str,
    description: str,
    cached_label: str,
    uncached_label: str,
    experiment_title: str,
    reverse_order: bool = False,
    request_delay: float = 2.0,
) -> Dict[str, Any]:
  """
  Run a cache performance comparison experiment for a specific model.

  Args:
      model_name: Model to test (e.g., "gemini-2.0-flash", "gemini-2.5-flash")
      description: Description of what the experiment tests
      cached_label: Label for the cached experiment variant
      uncached_label: Label for the uncached experiment variant
      experiment_title: Title to display for the experiment

  Returns:
      Dictionary containing experiment results and performance comparison
  """
  print("=" * 80)
  print(f"EXPERIMENT {model_name}: {experiment_title}")
  print("=" * 80)
  print(f"Testing: {description}")
  print(f"Model: {model_name}")
  print()

  # Create app variants
  app_cached = create_agent_variant(app, model_name, cache_enabled=True)
  app_uncached = create_agent_variant(app, model_name, cache_enabled=False)

  # Get test prompts
  prompts = get_test_prompts()

  # Create runners
  runner_cached = InMemoryRunner(app=app_cached, app_name=None)
  runner_uncached = InMemoryRunner(app=app_uncached, app_name=None)

  # Create sessions for each experiment to avoid cross-contamination
  session_cached = await runner_cached.session_service.create_session(
      app_name=runner_cached.app_name, user_id=USER_ID
  )
  session_uncached = await runner_uncached.session_service.create_session(
      app_name=runner_uncached.app_name, user_id=USER_ID
  )

  if not reverse_order:  # Default: uncached first
    print("▶️ Running experiments in DEFAULT ORDER (uncached first)")
    print()

    # Test uncached version first
    results_uncached = await run_experiment_batch(
        app_uncached.root_agent.name,
        runner_uncached,
        USER_ID,
        session_uncached.id,
        prompts,
        f"Experiment {model_name} - {uncached_label}",
        request_delay=request_delay,
    )

    # Brief pause between experiments
    await asyncio.sleep(5)

    # Test cached version second
    results_cached = await run_experiment_batch(
        app_cached.root_agent.name,
        runner_cached,
        USER_ID,
        session_cached.id,
        prompts,
        f"Experiment {model_name} - {cached_label}",
        request_delay=request_delay,
    )
  else:
    print("🔄 Running experiments in ALTERNATE ORDER (cached first)")
    print()

    # Test cached version first
    results_cached = await run_experiment_batch(
        app_cached.root_agent.name,
        runner_cached,
        USER_ID,
        session_cached.id,
        prompts,
        f"Experiment {model_name} - {cached_label}",
        request_delay=request_delay,
    )

    # Brief pause between experiments
    await asyncio.sleep(5)

    # Test uncached version second
    results_uncached = await run_experiment_batch(
        app_uncached.root_agent.name,
        runner_uncached,
        USER_ID,
        session_uncached.id,
        prompts,
        f"Experiment {model_name} - {uncached_label}",
        request_delay=request_delay,
    )

  # Analyze cache performance using CachePerformanceAnalyzer
  performance_analysis = await analyze_cache_performance_from_sessions(
      runner_cached,
      session_cached,
      runner_uncached,
      session_uncached,
      model_name,
  )

  # Extract metrics from analyzer for backward compatibility
  cached_analysis = performance_analysis.get("cached_analysis", {})
  uncached_analysis = performance_analysis.get("uncached_analysis", {})

  cached_total_prompt_tokens = cached_analysis.get("total_prompt_tokens", 0)
  cached_total_cached_tokens = cached_analysis.get("total_cached_tokens", 0)
  cached_cache_hit_ratio = cached_analysis.get("cache_hit_ratio_percent", 0.0)
  cached_cache_utilization_ratio = cached_analysis.get(
      "cache_utilization_ratio_percent", 0.0
  )
  cached_avg_cached_tokens_per_request = cached_analysis.get(
      "avg_cached_tokens_per_request", 0.0
  )
  cached_requests_with_hits = cached_analysis.get("requests_with_cache_hits", 0)
  total_cached_requests = cached_analysis.get("total_requests", 0)

  uncached_total_prompt_tokens = uncached_analysis.get("total_prompt_tokens", 0)
  uncached_total_cached_tokens = uncached_analysis.get("total_cached_tokens", 0)
  uncached_cache_hit_ratio = uncached_analysis.get(
      "cache_hit_ratio_percent", 0.0
  )
  uncached_cache_utilization_ratio = uncached_analysis.get(
      "cache_utilization_ratio_percent", 0.0
  )
  uncached_avg_cached_tokens_per_request = uncached_analysis.get(
      "avg_cached_tokens_per_request", 0.0
  )
  uncached_requests_with_hits = uncached_analysis.get(
      "requests_with_cache_hits", 0
  )
  total_uncached_requests = uncached_analysis.get("total_requests", 0)

  summary = {
      "experiment": model_name,
      "description": description,
      "model": model_name,
      "cached_results": results_cached,
      "uncached_results": results_uncached,
      "cache_analysis": {
          "cached_experiment": {
              "cache_hit_ratio_percent": cached_cache_hit_ratio,
              "cache_utilization_ratio_percent": cached_cache_utilization_ratio,
              "total_prompt_tokens": cached_total_prompt_tokens,
              "total_cached_tokens": cached_total_cached_tokens,
              "avg_cached_tokens_per_request": (
                  cached_avg_cached_tokens_per_request
              ),
              "requests_with_cache_hits": cached_requests_with_hits,
              "total_requests": total_cached_requests,
          },
          "uncached_experiment": {
              "cache_hit_ratio_percent": uncached_cache_hit_ratio,
              "cache_utilization_ratio_percent": (
                  uncached_cache_utilization_ratio
              ),
              "total_prompt_tokens": uncached_total_prompt_tokens,
              "total_cached_tokens": uncached_total_cached_tokens,
              "avg_cached_tokens_per_request": (
                  uncached_avg_cached_tokens_per_request
              ),
              "requests_with_cache_hits": uncached_requests_with_hits,
              "total_requests": total_uncached_requests,
          },
      },
  }

  print(f"📊 EXPERIMENT {model_name} CACHE ANALYSIS:")
  print(f"   🔥 {cached_label}:")
  print(
      f"      Cache Hit Ratio: {cached_cache_hit_ratio:.1f}%"
      f" ({cached_total_cached_tokens:,} /"
      f" {cached_total_prompt_tokens:,} tokens)"
  )
  print(
      f"      Cache Utilization: {cached_cache_utilization_ratio:.1f}%"
      f" ({cached_requests_with_hits}/{total_cached_requests} requests)"
  )
  print(
      "      Avg Cached Tokens/Request:"
      f" {cached_avg_cached_tokens_per_request:.0f}"
  )
  print(f"   ❄️  {uncached_label}:")
  print(
      f"      Cache Hit Ratio: {uncached_cache_hit_ratio:.1f}%"
      f" ({uncached_total_cached_tokens:,} /"
      f" {uncached_total_prompt_tokens:,} tokens)"
  )
  print(
      f"      Cache Utilization: {uncached_cache_utilization_ratio:.1f}%"
      f" ({uncached_requests_with_hits}/{total_uncached_requests} requests)"
  )
  print(
      "      Avg Cached Tokens/Request:"
      f" {uncached_avg_cached_tokens_per_request:.0f}"
  )
  print()

  # Add performance analysis to summary
  summary["performance_analysis"] = performance_analysis

  return summary


async def analyze_cache_performance_from_sessions(
    runner_cached,
    session_cached,
    runner_uncached,
    session_uncached,
    model_name: str,
) -> Dict[str, Any]:
  """Analyze cache performance using CachePerformanceAnalyzer."""
  print("📊 ANALYZING CACHE PERFORMANCE WITH CachePerformanceAnalyzer...")

  analyzer_cached = CachePerformanceAnalyzer(runner_cached.session_service)
  analyzer_uncached = CachePerformanceAnalyzer(runner_uncached.session_service)

  # Analyze cached experiment
  try:
    cached_analysis = await analyzer_cached.analyze_agent_cache_performance(
        session_cached.id,
        USER_ID,
        runner_cached.app_name,
        f"cache_analysis_{model_name.replace('.', '_').replace('-', '_')}_cached",
    )
    print(f"  🔥 Cached Experiment Analysis:")
    print(f"     Status: {cached_analysis['status']}")
    if cached_analysis["status"] == "active":
      print(
          "     Cache Hit Ratio:"
          f" {cached_analysis['cache_hit_ratio_percent']:.1f}%"
          f" ({cached_analysis['total_cached_tokens']:,} /"
          f" {cached_analysis['total_prompt_tokens']:,} tokens)"
      )
      print(
          "     Cache Utilization:"
          f" {cached_analysis['cache_utilization_ratio_percent']:.1f}%"
          f" ({cached_analysis['requests_with_cache_hits']}/{cached_analysis['total_requests']} requests)"
      )
      print(
          "     Avg Cached Tokens/Request:"
          f" {cached_analysis['avg_cached_tokens_per_request']:.0f}"
      )
      print(
          f"     Requests with cache: {cached_analysis['requests_with_cache']}"
      )
      print(
          "     Avg invocations used:"
          f" {cached_analysis['avg_invocations_used']:.1f}"
      )
      print(f"     Cache refreshes: {cached_analysis['cache_refreshes']}")
      print(f"     Total invocations: {cached_analysis['total_invocations']}")
  except Exception as e:
    print(f"     ❌ Error analyzing cached experiment: {e}")
    cached_analysis = {"status": "error", "error": str(e)}

  # Analyze uncached experiment
  try:
    uncached_analysis = await analyzer_uncached.analyze_agent_cache_performance(
        session_uncached.id,
        USER_ID,
        runner_uncached.app_name,
        f"cache_analysis_{model_name.replace('.', '_').replace('-', '_')}_no_cache",
    )
    print(f"  ❄️  Uncached Experiment Analysis:")
    print(f"     Status: {uncached_analysis['status']}")
    if uncached_analysis["status"] == "active":
      print(
          "     Cache Hit Ratio:"
          f" {uncached_analysis['cache_hit_ratio_percent']:.1f}%"
          f" ({uncached_analysis['total_cached_tokens']:,} /"
          f" {uncached_analysis['total_prompt_tokens']:,} tokens)"
      )
      print(
          "     Cache Utilization:"
          f" {uncached_analysis['cache_utilization_ratio_percent']:.1f}%"
          f" ({uncached_analysis['requests_with_cache_hits']}/{uncached_analysis['total_requests']} requests)"
      )
      print(
          "     Avg Cached Tokens/Request:"
          f" {uncached_analysis['avg_cached_tokens_per_request']:.0f}"
      )
      print(
          "     Requests with cache:"
          f" {uncached_analysis['requests_with_cache']}"
      )
      print(
          "     Avg invocations used:"
          f" {uncached_analysis['avg_invocations_used']:.1f}"
      )
      print(f"     Cache refreshes: {uncached_analysis['cache_refreshes']}")
      print(f"     Total invocations: {uncached_analysis['total_invocations']}")
  except Exception as e:
    print(f"     ❌ Error analyzing uncached experiment: {e}")
    uncached_analysis = {"status": "error", "error": str(e)}

  print()

  return {
      "cached_analysis": cached_analysis,
      "uncached_analysis": uncached_analysis,
  }


def get_experiment_labels(model_name: str) -> Dict[str, str]:
  """Get experiment labels and titles for a given model."""
  # Determine experiment type based on model name
  if "2.5" in model_name:
    # Gemini 2.5 models have implicit caching
    return {
        "description": "Google implicit caching vs ADK explicit caching",
        "cached_label": "Explicit Caching",
        "uncached_label": "Implicit Caching",
        "experiment_title": "Implicit vs Explicit Caching",
    }
  else:
    # Other models (2.0, etc.) test explicit caching vs no caching
    return {
        "description": "ADK explicit caching enabled vs disabled",
        "cached_label": "Cached",
        "uncached_label": "Uncached",
        "experiment_title": "Cache Performance Comparison",
    }


def calculate_averaged_results(
    all_results: List[Dict[str, Any]], model_name: str
) -> Dict[str, Any]:
  """Calculate averaged results from multiple experiment runs."""
  if not all_results:
    raise ValueError("No results to average")

  # Calculate average cache metrics
  cache_hit_ratios = [
      r["cache_analysis"]["cache_hit_ratio_percent"] for r in all_results
  ]
  cache_utilization_ratios = [
      r["cache_analysis"]["cache_utilization_ratio_percent"]
      for r in all_results
  ]
  total_prompt_tokens = [
      r["cache_analysis"]["total_prompt_tokens"] for r in all_results
  ]
  total_cached_tokens = [
      r["cache_analysis"]["total_cached_tokens"] for r in all_results
  ]
  avg_cached_tokens_per_request = [
      r["cache_analysis"]["avg_cached_tokens_per_request"] for r in all_results
  ]
  requests_with_cache_hits = [
      r["cache_analysis"]["requests_with_cache_hits"] for r in all_results
  ]

  def safe_average(values):
    """Calculate average, handling empty lists."""
    return sum(values) / len(values) if values else 0.0

  # Create averaged result
  averaged_result = {
      "experiment": model_name,
      "description": all_results[0]["description"],
      "model": model_name,
      "individual_runs": (
          all_results
      ),  # Keep all individual results for reference
      "averaged_cache_analysis": {
          "cache_hit_ratio_percent": safe_average(cache_hit_ratios),
          "cache_utilization_ratio_percent": safe_average(
              cache_utilization_ratios
          ),
          "total_prompt_tokens": safe_average(total_prompt_tokens),
          "total_cached_tokens": safe_average(total_cached_tokens),
          "avg_cached_tokens_per_request": safe_average(
              avg_cached_tokens_per_request
          ),
          "requests_with_cache_hits": safe_average(requests_with_cache_hits),
      },
      "statistics": {
          "runs_completed": len(all_results),
          "cache_hit_ratio_std": _calculate_std(cache_hit_ratios),
          "cache_utilization_std": _calculate_std(cache_utilization_ratios),
          "cached_tokens_per_request_std": _calculate_std(
              avg_cached_tokens_per_request
          ),
      },
  }

  # Print averaged results
  print("\n📊 AVERAGED CACHE ANALYSIS RESULTS:")
  print("=" * 80)
  avg_cache = averaged_result["averaged_cache_analysis"]
  stats = averaged_result["statistics"]

  print(f"   Runs completed: {stats['runs_completed']}")
  print(
      f"   Average Cache Hit Ratio: {avg_cache['cache_hit_ratio_percent']:.1f}%"
      f" (±{stats['cache_hit_ratio_std']:.1f}%)"
  )
  print(
      "   Average Cache Utilization:"
      f" {avg_cache['cache_utilization_ratio_percent']:.1f}%"
      f" (±{stats['cache_utilization_std']:.1f}%)"
  )
  print(
      "   Average Cached Tokens/Request:"
      f" {avg_cache['avg_cached_tokens_per_request']:.0f}"
      f" (±{stats['cached_tokens_per_request_std']:.0f})"
  )
  print()

  return averaged_result


def _calculate_std(values):
  """Calculate standard deviation."""
  if len(values) <= 1:
    return 0.0
  mean = sum(values) / len(values)
  variance = sum((x - mean) ** 2 for x in values) / len(values)
  return variance**0.5


def save_results(results: Dict[str, Any], filename: str):
  """Save experiment results to JSON file."""
  with open(filename, "w") as f:
    json.dump(results, f, indent=2)
  print(f"💾 Results saved to: {filename}")


async def main():
  """Run cache performance experiment for a specific model."""
  parser = argparse.ArgumentParser(
      description="ADK Cache Performance Experiment"
  )
  parser.add_argument(
      "model",
      help="Model to test (e.g., gemini-2.5-flash, gemini-2.0-flash-001)",
  )
  parser.add_argument(
      "--output",
      help="Output filename for results (default: cache_{model}_results.json)",
  )
  parser.add_argument(
      "--repeat",
      type=int,
      default=1,
      help=(
          "Number of times to repeat each experiment for averaged results"
          " (default: 1)"
      ),
  )
  parser.add_argument(
      "--cached-first",
      action="store_true",
      help="Run cached experiment first (default: uncached first)",
  )
  parser.add_argument(
      "--request-delay",
      type=float,
      default=2.0,
      help=(
          "Delay in seconds between API requests to avoid overloading (default:"
          " 2.0)"
      ),
  )
  parser.add_argument(
      "--log-level",
      choices=["DEBUG", "INFO", "WARNING", "ERROR"],
      default="INFO",
      help="Set logging level (default: INFO)",
  )

  args = parser.parse_args()

  # Setup logger with specified level
  log_level = getattr(logging, args.log_level.upper())
  logs.setup_adk_logger(log_level)

  # Set default output filename based on model
  if not args.output:
    args.output = (
        f"cache_{args.model.replace('.', '_').replace('-', '_')}_results.json"
    )

  print("🧪 ADK CONTEXT CACHE PERFORMANCE EXPERIMENT")
  print("=" * 80)
  print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
  print(f"Model: {args.model}")
  print(f"Repetitions: {args.repeat}")
  print()

  start_time = time.time()

  try:
    # Get experiment labels for the model
    labels = get_experiment_labels(args.model)

    # Run the experiment multiple times if repeat > 1
    if args.repeat == 1:
      # Single run
      result = await run_cache_comparison_experiment(
          model_name=args.model,
          reverse_order=args.cached_first,
          request_delay=args.request_delay,
          **labels,
      )
    else:
      # Multiple runs with averaging
      print(f"🔄 Running experiment {args.repeat} times for averaged results")
      print("=" * 80)

      all_results = []
      for run_num in range(args.repeat):
        print(f"\n🏃 RUN {run_num + 1}/{args.repeat}")
        print("-" * 40)

        run_result = await run_cache_comparison_experiment(
            model_name=args.model,
            reverse_order=args.cached_first,
            request_delay=args.request_delay,
            **labels,
        )
        all_results.append(run_result)

        # Brief pause between runs
        if run_num < args.repeat - 1:
          print("⏸️  Pausing 10 seconds between runs...")
          await asyncio.sleep(10)

      # Calculate averaged results
      result = calculate_averaged_results(all_results, args.model)

    # Add completion metadata
    result["end_time"] = time.strftime("%Y-%m-%d %H:%M:%S")
    result["total_duration"] = time.time() - start_time
    result["repetitions"] = args.repeat

  except KeyboardInterrupt:
    print("\n⚠️ Experiment interrupted by user")
    sys.exit(1)
  except Exception as e:
    print(f"\n❌ Experiment failed: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

  # Save results
  save_results(result, args.output)

  # Print final summary
  print("=" * 80)
  print("🎉 EXPERIMENT COMPLETED SUCCESSFULLY!")
  print("=" * 80)

  # Handle both single and averaged results
  if args.repeat == 1:
    cached_exp = result["cache_analysis"]["cached_experiment"]
    uncached_exp = result["cache_analysis"]["uncached_experiment"]
    labels = get_experiment_labels(args.model)
    print(f"{args.model}:")
    print(f"  🔥 {labels['cached_label']}:")
    print(f"    Cache Hit Ratio: {cached_exp['cache_hit_ratio_percent']:.1f}%")
    print(
        "    Cache Utilization:"
        f" {cached_exp['cache_utilization_ratio_percent']:.1f}%"
    )
    print(
        "    Cached Tokens/Request:"
        f" {cached_exp['avg_cached_tokens_per_request']:.0f}"
    )
    print(f"  ❄️  {labels['uncached_label']}:")
    print(
        f"    Cache Hit Ratio: {uncached_exp['cache_hit_ratio_percent']:.1f}%"
    )
    print(
        "    Cache Utilization:"
        f" {uncached_exp['cache_utilization_ratio_percent']:.1f}%"
    )
    print(
        "    Cached Tokens/Request:"
        f" {uncached_exp['avg_cached_tokens_per_request']:.0f}"
    )
  else:
    # For averaged results, show summary comparison
    cached_exp = result["averaged_cache_analysis"]["cached_experiment"]
    uncached_exp = result["averaged_cache_analysis"]["uncached_experiment"]
    labels = get_experiment_labels(args.model)
    print(f"{args.model} (averaged over {args.repeat} runs):")
    print(f"  🔥 {labels['cached_label']} vs ❄️  {labels['uncached_label']}:")
    print(
        f"    Cache Hit Ratio: {cached_exp['cache_hit_ratio_percent']:.1f}% vs"
        f" {uncached_exp['cache_hit_ratio_percent']:.1f}%"
    )
    print(
        "    Cache Utilization:"
        f" {cached_exp['cache_utilization_ratio_percent']:.1f}% vs"
        f" {uncached_exp['cache_utilization_ratio_percent']:.1f}%"
    )

  print(f"\nTotal execution time: {result['total_duration']:.2f} seconds")
  print(f"Results saved to: {args.output}")


if __name__ == "__main__":
  asyncio.run(main())
