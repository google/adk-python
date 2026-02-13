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

"""Utility functions for cache analysis experiments."""

import asyncio
import time
from typing import Any
from typing import Dict
from typing import List

from google.adk.runners import InMemoryRunner


async def call_agent_async(
    runner: InMemoryRunner, user_id: str, session_id: str, prompt: str
) -> Dict[str, Any]:
  """Call agent asynchronously and return response with token usage."""
  from google.genai import types

  response_parts = []
  token_usage = {
      "prompt_token_count": 0,
      "candidates_token_count": 0,
      "cached_content_token_count": 0,
      "total_token_count": 0,
  }

  async for event in runner.run_async(
      user_id=user_id,
      session_id=session_id,
      new_message=types.Content(parts=[types.Part(text=prompt)], role="user"),
  ):
    if event.content and event.content.parts:
      for part in event.content.parts:
        if hasattr(part, "text") and part.text:
          response_parts.append(part.text)

    # Collect token usage information
    if event.usage_metadata:
      if (
          hasattr(event.usage_metadata, "prompt_token_count")
          and event.usage_metadata.prompt_token_count
      ):
        token_usage[
            "prompt_token_count"
        ] += event.usage_metadata.prompt_token_count
      if (
          hasattr(event.usage_metadata, "candidates_token_count")
          and event.usage_metadata.candidates_token_count
      ):
        token_usage[
            "candidates_token_count"
        ] += event.usage_metadata.candidates_token_count
      if (
          hasattr(event.usage_metadata, "cached_content_token_count")
          and event.usage_metadata.cached_content_token_count
      ):
        token_usage[
            "cached_content_token_count"
        ] += event.usage_metadata.cached_content_token_count
      if (
          hasattr(event.usage_metadata, "total_token_count")
          and event.usage_metadata.total_token_count
      ):
        token_usage[
            "total_token_count"
        ] += event.usage_metadata.total_token_count

  response_text = "".join(response_parts)

  return {"response_text": response_text, "token_usage": token_usage}


def get_test_prompts() -> List[str]:
  """Get a standardized set of test prompts for cache analysis experiments.

  Designed for consistent behavior:
  - Prompts 1-5: Will NOT trigger function calls (general questions)
  - Prompts 6-10: Will trigger function calls (specific tool requests)
  """
  return [
      # === PROMPTS THAT WILL NOT TRIGGER FUNCTION CALLS ===
      # (General questions that don't match specific tool descriptions)
      "Hello, what can you do for me?",
      (
          "What is artificial intelligence and how does it work in modern"
          " applications?"
      ),
      "Explain the difference between machine learning and deep learning.",
      "What are the main challenges in implementing AI systems at scale?",
      "How do recommendation systems work in modern e-commerce platforms?",
      # === PROMPTS THAT WILL TRIGGER FUNCTION CALLS ===
      # (Specific requests with all required parameters clearly specified)
      (
          "Use benchmark_performance with system_name='E-commerce Platform',"
          " metrics=['latency', 'throughput'], duration='standard',"
          " load_profile='realistic'."
      ),
      (
          "Call analyze_user_behavior_patterns with"
          " user_segment='premium_customers', time_period='last_30_days',"
          " metrics=['engagement', 'conversion']."
      ),
      (
          "Run market_research_analysis for industry='fintech',"
          " focus_areas=['user_experience', 'security'],"
          " report_depth='comprehensive'."
      ),
      (
          "Execute competitive_analysis with competitors=['Netflix',"
          " 'Disney+'], analysis_type='feature_comparison',"
          " output_format='detailed'."
      ),
      (
          "Perform content_performance_evaluation on content_type='video',"
          " platform='social_media', success_metrics=['views', 'engagement']."
      ),
  ]


async def run_experiment_batch(
    agent_name: str,
    runner: InMemoryRunner,
    user_id: str,
    session_id: str,
    prompts: List[str],
    experiment_name: str,
    request_delay: float = 2.0,
) -> Dict[str, Any]:
  """Run a batch of prompts and collect cache metrics."""
  results = []

  print(f"🧪 Running {experiment_name}")
  print(f"Agent: {agent_name}")
  print(f"Session: {session_id}")
  print(f"Prompts: {len(prompts)}")
  print(f"Request delay: {request_delay}s between calls")
  print("-" * 60)

  for i, prompt in enumerate(prompts, 1):
    print(f"[{i}/{len(prompts)}] Running test prompt...")
    print(f"Prompt: {prompt[:100]}...")

    try:
      agent_response = await call_agent_async(
          runner, user_id, session_id, prompt
      )

      result = {
          "prompt_number": i,
          "prompt": prompt,
          "response_length": len(agent_response["response_text"]),
          "success": True,
          "error": None,
          "token_usage": agent_response["token_usage"],
      }

      # Extract token usage for individual prompt statistics
      prompt_tokens = agent_response["token_usage"].get("prompt_token_count", 0)
      cached_tokens = agent_response["token_usage"].get(
          "cached_content_token_count", 0
      )

      print(
          "✅ Completed (Response:"
          f" {len(agent_response['response_text'])} chars)"
      )
      print(
          f"   📊 Tokens - Prompt: {prompt_tokens:,}, Cached: {cached_tokens:,}"
      )

    except Exception as e:
      result = {
          "prompt_number": i,
          "prompt": prompt,
          "response_length": 0,
          "success": False,
          "error": str(e),
          "token_usage": {
              "prompt_token_count": 0,
              "candidates_token_count": 0,
              "cached_content_token_count": 0,
              "total_token_count": 0,
          },
      }

      print(f"❌ Failed: {e}")

    results.append(result)

    # Configurable pause between requests to avoid API overload
    if i < len(prompts):  # Don't sleep after the last request
      print(f"   ⏸️  Waiting {request_delay}s before next request...")
      await asyncio.sleep(request_delay)

  successful_requests = sum(1 for r in results if r["success"])

  # Calculate cache statistics for this batch
  total_prompt_tokens = sum(
      r.get("token_usage", {}).get("prompt_token_count", 0) for r in results
  )
  total_cached_tokens = sum(
      r.get("token_usage", {}).get("cached_content_token_count", 0)
      for r in results
  )

  # Calculate cache hit ratio
  if total_prompt_tokens > 0:
    cache_hit_ratio = (total_cached_tokens / total_prompt_tokens) * 100
  else:
    cache_hit_ratio = 0.0

  # Calculate cache utilization
  requests_with_cache_hits = sum(
      1
      for r in results
      if r.get("token_usage", {}).get("cached_content_token_count", 0) > 0
  )
  cache_utilization_ratio = (
      (requests_with_cache_hits / len(prompts)) * 100 if prompts else 0.0
  )

  # Average cached tokens per request
  avg_cached_tokens_per_request = (
      total_cached_tokens / len(prompts) if prompts else 0.0
  )

  summary = {
      "experiment_name": experiment_name,
      "agent_name": agent_name,
      "total_requests": len(prompts),
      "successful_requests": successful_requests,
      "results": results,
      "cache_statistics": {
          "cache_hit_ratio_percent": cache_hit_ratio,
          "cache_utilization_ratio_percent": cache_utilization_ratio,
          "total_prompt_tokens": total_prompt_tokens,
          "total_cached_tokens": total_cached_tokens,
          "avg_cached_tokens_per_request": avg_cached_tokens_per_request,
          "requests_with_cache_hits": requests_with_cache_hits,
      },
  }

  print("-" * 60)
  print(f"✅ {experiment_name} completed:")
  print(f"   Total requests: {len(prompts)}")
  print(f"   Successful: {successful_requests}/{len(prompts)}")
  print("   📊 BATCH CACHE STATISTICS:")
  print(
      f"      Cache Hit Ratio: {cache_hit_ratio:.1f}%"
      f" ({total_cached_tokens:,} / {total_prompt_tokens:,} tokens)"
  )
  print(
      f"      Cache Utilization: {cache_utilization_ratio:.1f}%"
      f" ({requests_with_cache_hits}/{len(prompts)} requests)"
  )
  print(f"      Avg Cached Tokens/Request: {avg_cached_tokens_per_request:.0f}")
  print()

  return summary
