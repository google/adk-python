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

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime
from datetime import timedelta
import logging
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
      cache_duration: timedelta = timedelta(hours=24),
  ):
    """Initialize the pricing service.

    Args:
      pricing_url: URL to fetch pricing information from.
      cache_duration: How long to cache pricing data before refreshing.
    """
    self._pricing_url = pricing_url
    self._cache_duration = cache_duration
    self._cached_pricing: dict[str, ModelPricing] = _DEFAULT_MODEL_PRICING
    self._last_updated: Optional[datetime] = None
    self._fetch_lock = asyncio.Lock()

  async def get_pricing(self, model_name: str) -> Optional[ModelPricing]:
    """Get pricing for a specific model.

    Args:
      model_name: Name of the Gemini model (e.g., "gemini-2.5-flash").

    Returns:
      ModelPricing object if found, None otherwise.
    """
    # Normalize model name (remove prefixes like "models/")
    normalized_name = model_name.split('/')[-1]

    # Check if we need to refresh the cache
    if self._should_refresh_cache():
      await self._refresh_pricing()

    # Try to find exact match
    if normalized_name in self._cached_pricing:
      return self._cached_pricing[normalized_name]

    # Try to find fuzzy match (e.g., "gemini-2.5-flash-001" -> "gemini-2.5-flash")
    for key in self._cached_pricing:
      if normalized_name.startswith(key):
        return self._cached_pricing[key]

    _logger.warning(
        'Pricing not found for model: %s, using default', model_name
    )
    return None

  def _should_refresh_cache(self) -> bool:
    """Check if the pricing cache should be refreshed."""
    if self._last_updated is None:
      return False  # Use defaults on first run
    return datetime.now() - self._last_updated > self._cache_duration

  async def _refresh_pricing(self) -> None:
    """Refresh pricing data from the Vertex AI pricing page.

    Note: This is a placeholder implementation. In production, you would
    either parse the pricing page HTML or use an official API if available.
    For now, we use the hardcoded defaults.
    """
    async with self._fetch_lock:
      # Double-check to avoid race conditions
      if not self._should_refresh_cache():
        return

      try:
        # TODO: Implement actual pricing page parsing or API call
        # For now, we just use the hardcoded defaults
        _logger.info('Using default Gemini pricing (no dynamic fetch yet)')
        self._last_updated = datetime.now()
      except Exception as e:
        _logger.error('Failed to refresh Gemini pricing: %s', e)


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
