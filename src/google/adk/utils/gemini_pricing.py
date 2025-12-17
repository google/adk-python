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

"""Gemini API pricing calculator with live pricing fetching.

This module provides utilities to calculate token costs for Gemini models.
On first use, it attempts to fetch the latest pricing from Google Cloud's
pricing page and caches it permanently for the session. If fetching fails,
it falls back to hardcoded defaults (accurate as of December 2025).

Features:
  - Automatic pricing fetch from cloud.google.com/vertex-ai/generative-ai/pricing
  - One-time fetch on first request, then cached permanently
  - Fallback to hardcoded defaults only if fetching fails
  - Support for tiered pricing (low/high token thresholds)
  - Cached token pricing calculation
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
import logging
import re
from typing import Optional

import aiohttp

_logger = logging.getLogger(__name__)


@dataclass
class ModelPricing:
  """Pricing information for a specific Gemini model.

  All prices are in USD per 1 million tokens.
  """

  input_price_low: float
  input_price_high: float
  output_price_low: float
  output_price_high: float
  cached_input_price_low: float
  cached_input_price_high: float
  threshold_tokens: int = 200_000

  def calculate_cost(
      self,
      prompt_tokens: int,
      output_tokens: int,
      cached_tokens: int = 0,
  ) -> float:
    """Calculate the total cost for a request.

    Args:
      prompt_tokens: Number of prompt tokens (excluding cached tokens).
      output_tokens: Number of output tokens generated.
      cached_tokens: Number of cached prompt tokens.

    Returns:
      Total cost in USD.
    """
    total_input_tokens = prompt_tokens + cached_tokens

    # Determine if we're in the high-tier pricing
    use_high_tier = total_input_tokens > self.threshold_tokens

    # Calculate input cost (non-cached)
    input_price = (
        self.input_price_high if use_high_tier else self.input_price_low
    )
    input_cost = (prompt_tokens / 1_000_000) * input_price

    # Calculate cached input cost
    cached_price = (
        self.cached_input_price_high
        if use_high_tier
        else self.cached_input_price_low
    )
    cached_cost = (cached_tokens / 1_000_000) * cached_price

    # Calculate output cost
    output_price = (
        self.output_price_high if use_high_tier else self.output_price_low
    )
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + cached_cost + output_cost


# Default pricing for common Gemini models (fallback if fetching fails)
# Prices are per 1 million tokens in USD
# Last updated: December 2025 from https://cloud.google.com/vertex-ai/generative-ai/pricing
_DEFAULT_MODEL_PRICING = {
    'gemini-2.5-pro': ModelPricing(
        input_price_low=1.25,
        input_price_high=2.50,
        output_price_low=10.00,
        output_price_high=15.00,
        cached_input_price_low=0.125,
        cached_input_price_high=0.250,
    ),
    'gemini-2.5-flash': ModelPricing(
        input_price_low=0.30,
        input_price_high=0.30,
        output_price_low=2.50,
        output_price_high=2.50,
        cached_input_price_low=0.030,
        cached_input_price_high=0.030,
    ),
    'gemini-2.5-flash-lite': ModelPricing(
        input_price_low=0.10,
        input_price_high=0.10,
        output_price_low=0.40,
        output_price_high=0.40,
        cached_input_price_low=0.010,
        cached_input_price_high=0.010,
    ),
    'gemini-2.0-flash': ModelPricing(
        input_price_low=0.15,
        input_price_high=0.15,
        output_price_low=0.60,
        output_price_high=0.60,
        cached_input_price_low=0.015,
        cached_input_price_high=0.015,
    ),
    'gemini-2.0-flash-lite': ModelPricing(
        input_price_low=0.075,
        input_price_high=0.075,
        output_price_low=0.30,
        output_price_high=0.30,
        cached_input_price_low=0.0075,
        cached_input_price_high=0.0075,
    ),
    'gemini-1.5-pro': ModelPricing(
        input_price_low=1.25,
        input_price_high=2.50,
        output_price_low=5.00,
        output_price_high=10.00,
        cached_input_price_low=0.3125,
        cached_input_price_high=0.625,
        threshold_tokens=128_000,
    ),
    'gemini-1.5-flash': ModelPricing(
        input_price_low=0.075,
        input_price_high=0.15,
        output_price_low=0.30,
        output_price_high=0.60,
        cached_input_price_low=0.01875,
        cached_input_price_high=0.0375,
        threshold_tokens=128_000,
    ),
}


class GeminiPricingService:
  """Service for fetching and caching Gemini API pricing information."""

  def __init__(
      self,
      pricing_url: str = (
          'https://cloud.google.com/vertex-ai/generative-ai/pricing'
      ),
      enable_fetch: bool = True,
  ):
    """Initialize the pricing service.

    Args:
      pricing_url: URL to fetch pricing information from.
      enable_fetch: If False, skip fetching and use hardcoded defaults only.
                    Useful for testing.
    """
    self._pricing_url = pricing_url
    self._enable_fetch = enable_fetch
    self._cached_pricing: Optional[dict[str, ModelPricing]] = None
    self._fetch_attempted: bool = False
    self._fetch_lock = asyncio.Lock()

  async def get_pricing(self, model_name: str) -> Optional[ModelPricing]:
    """Get pricing for a specific model.

    Fetches pricing from Google Cloud on first call, then caches permanently.
    Falls back to hardcoded defaults only if fetching fails or is disabled.

    Args:
      model_name: Name of the Gemini model (e.g., "gemini-2.5-flash").

    Returns:
      ModelPricing object if found, None otherwise.
    """
    # Fetch pricing on first call (if enabled)
    if self._enable_fetch and not self._fetch_attempted:
      await self._refresh_pricing()

    # Normalize model name (remove prefixes like "models/")
    normalized_name = model_name.split('/')[-1]

    # Use cached pricing (either fetched or fallback)
    pricing_data = self._cached_pricing or _DEFAULT_MODEL_PRICING

    # Try to find exact match
    if normalized_name in pricing_data:
      return pricing_data[normalized_name]

    # Try to find fuzzy match (e.g., "gemini-2.5-flash-001" -> "gemini-2.5-flash")
    for key in pricing_data:
      if normalized_name.startswith(key):
        return pricing_data[key]

    _logger.warning(
        'Pricing not found for model: %s, using default', model_name
    )
    return None

  async def _refresh_pricing(self) -> None:
    """Fetch pricing data from the Vertex AI pricing page on first call.

    Attempts to fetch and parse the latest pricing from Google Cloud's
    pricing page. Falls back to hardcoded defaults only if fetching fails.
    This is called only once - on the first pricing request.
    """
    async with self._fetch_lock:
      # Double-check to avoid race conditions
      if self._fetch_attempted:
        return

      self._fetch_attempted = True

      try:
        _logger.info(
            'Fetching latest Gemini pricing from %s', self._pricing_url
        )

        async with aiohttp.ClientSession() as session:
          async with session.get(
              self._pricing_url, timeout=aiohttp.ClientTimeout(total=10)
          ) as response:
            if response.status != 200:
              _logger.warning(
                  'Failed to fetch pricing page (status %d), using hardcoded'
                  ' defaults',
                  response.status,
              )
              self._cached_pricing = _DEFAULT_MODEL_PRICING
              return

            html_content = await response.text()
            parsed_pricing = self._parse_pricing_page(html_content)

            if parsed_pricing:
              # Merge parsed pricing with defaults (in case some models are missing)
              self._cached_pricing = {
                  **_DEFAULT_MODEL_PRICING,
                  **parsed_pricing,
              }
              _logger.info(
                  'Successfully fetched pricing for %d models from API',
                  len(parsed_pricing),
              )
            else:
              _logger.warning(
                  'No pricing found in API response, using hardcoded defaults'
              )
              self._cached_pricing = _DEFAULT_MODEL_PRICING

      except Exception as e:
        _logger.warning(
            'Failed to fetch Gemini pricing: %s, using hardcoded defaults', e
        )
        self._cached_pricing = _DEFAULT_MODEL_PRICING

  def _parse_pricing_page(self, html_content: str) -> dict[str, ModelPricing]:
    """Parse pricing information from the HTML page.

    Args:
      html_content: HTML content of the pricing page.

    Returns:
      Dictionary mapping model names to ModelPricing objects.
      Returns empty dict if parsing fails or produces invalid results.
    """
    pricing_data = {}

    try:
      # Look for pricing tables in the HTML
      # The pricing page typically has tables with model names and prices
      # Pattern: Match prices in format like "$0.30" or "$1.25"
      price_pattern = r'\$(\d+\.?\d*)'

      # Try to find Gemini model sections and their associated prices
      # This is a best-effort parsing and may need updates if the page structure changes

      # Look for common model names in the content
      model_patterns = {
          'gemini-2.5-flash': r'Gemini 2\.5 Flash',
          'gemini-2.5-pro': r'Gemini 2\.5 Pro',
          'gemini-2.0-flash': r'Gemini 2\.0 Flash',
          'gemini-1.5-pro': r'Gemini 1\.5 Pro',
          'gemini-1.5-flash': r'Gemini 1\.5 Flash',
      }

      for model_key, model_pattern in model_patterns.items():
        match = re.search(model_pattern, html_content, re.IGNORECASE)
        if match:
          # Find the section containing this model
          section_start = match.start()
          section_end = min(section_start + 5000, len(html_content))
          section = html_content[section_start:section_end]

          # Extract all prices in this section
          prices = re.findall(price_pattern, section)

          if len(prices) >= 2:
            # Typically: input_low, input_high, output_low, output_high
            # or just: input, output (if no tiering)
            try:
              input_low = float(prices[0])
              output_low = float(prices[1]) if len(prices) > 1 else input_low

              # Check if there's tiered pricing
              input_high = float(prices[2]) if len(prices) > 2 else input_low
              output_high = float(prices[3]) if len(prices) > 3 else output_low

              # Validate pricing - sanity check to avoid garbage data
              # Gemini prices should be < $100 per 1M tokens
              if (
                  input_low > 100
                  or input_high > 100
                  or output_low > 100
                  or output_high > 100
              ):
                _logger.warning(
                    'Parsed pricing for %s looks invalid (>$100/1M tokens),'
                    ' skipping',
                    model_key,
                )
                continue

              # Cached pricing is typically 10% of regular pricing
              cached_low = input_low * 0.1
              cached_high = input_high * 0.1

              pricing_data[model_key] = ModelPricing(
                  input_price_low=input_low,
                  input_price_high=input_high,
                  output_price_low=output_low,
                  output_price_high=output_high,
                  cached_input_price_low=cached_low,
                  cached_input_price_high=cached_high,
              )
              _logger.debug(
                  'Parsed pricing for %s: in=$%.2f-$%.2f, out=$%.2f-$%.2f',
                  model_key,
                  input_low,
                  input_high,
                  output_low,
                  output_high,
              )
            except (ValueError, IndexError) as e:
              _logger.debug('Failed to parse prices for %s: %s', model_key, e)
              continue

    except Exception as e:
      _logger.warning('Error parsing pricing page: %s', e)

    return pricing_data


# Global pricing service instance
_pricing_service: Optional[GeminiPricingService] = None


def get_pricing_service() -> GeminiPricingService:
  """Get the global pricing service instance."""
  global _pricing_service
  if _pricing_service is None:
    _pricing_service = GeminiPricingService()
  return _pricing_service


async def calculate_token_cost(
    model_name: str,
    prompt_tokens: int,
    output_tokens: int,
    cached_tokens: int = 0,
) -> Optional[float]:
  """Calculate the cost of a model invocation.

  Args:
    model_name: Name of the Gemini model.
    prompt_tokens: Number of prompt tokens.
    output_tokens: Number of output tokens.
    cached_tokens: Number of cached tokens.

  Returns:
    Total cost in USD, or None if pricing not available.
  """
  service = get_pricing_service()
  pricing = await service.get_pricing(model_name)

  if pricing is None:
    return None

  return pricing.calculate_cost(prompt_tokens, output_tokens, cached_tokens)
