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

from google.adk.utils.gemini_pricing import calculate_token_cost
from google.adk.utils.gemini_pricing import GeminiPricingService
from google.adk.utils.gemini_pricing import ModelPricing
import pytest


class TestModelPricing:
  """Test the ModelPricing class."""

  def test_calculate_cost_low_tier(self):
    """Test cost calculation for low-tier usage."""
    pricing = ModelPricing(
        input_price_low=1.25,
        input_price_high=2.50,
        output_price_low=10.00,
        output_price_high=15.00,
        cached_input_price_low=0.125,
        cached_input_price_high=0.250,
        threshold_tokens=200_000,
    )

    # Test with 1000 prompt tokens, 500 output tokens, no cache
    cost = pricing.calculate_cost(1000, 500, 0)
    expected = (1000 / 1_000_000) * 1.25 + (500 / 1_000_000) * 10.00
    assert abs(cost - expected) < 0.000001

  def test_calculate_cost_high_tier(self):
    """Test cost calculation for high-tier usage (>200K tokens)."""
    pricing = ModelPricing(
        input_price_low=1.25,
        input_price_high=2.50,
        output_price_low=10.00,
        output_price_high=15.00,
        cached_input_price_low=0.125,
        cached_input_price_high=0.250,
        threshold_tokens=200_000,
    )

    # Test with 250K prompt tokens, 50K output tokens
    cost = pricing.calculate_cost(250_000, 50_000, 0)
    expected = (250_000 / 1_000_000) * 2.50 + (50_000 / 1_000_000) * 15.00
    assert abs(cost - expected) < 0.000001

  def test_calculate_cost_with_cache(self):
    """Test cost calculation with cached tokens."""
    pricing = ModelPricing(
        input_price_low=1.25,
        input_price_high=2.50,
        output_price_low=10.00,
        output_price_high=15.00,
        cached_input_price_low=0.125,
        cached_input_price_high=0.250,
        threshold_tokens=200_000,
    )

    # Test with 1000 prompt tokens, 500 output tokens, 5000 cached tokens
    cost = pricing.calculate_cost(1000, 500, 5000)
    expected = (
        (1000 / 1_000_000) * 1.25
        + (5000 / 1_000_000) * 0.125
        + (500 / 1_000_000) * 10.00
    )
    assert abs(cost - expected) < 0.000001

  def test_calculate_cost_flash_model(self):
    """Test cost calculation for Flash model."""
    pricing = ModelPricing(
        input_price_low=0.30,
        input_price_high=0.30,
        output_price_low=2.50,
        output_price_high=2.50,
        cached_input_price_low=0.030,
        cached_input_price_high=0.030,
    )

    # Test with 10000 prompt tokens, 5000 output tokens
    cost = pricing.calculate_cost(10_000, 5_000, 0)
    expected = (10_000 / 1_000_000) * 0.30 + (5_000 / 1_000_000) * 2.50
    assert abs(cost - expected) < 0.000001


class TestGeminiPricingService:
  """Test the GeminiPricingService class."""

  @pytest.mark.asyncio
  async def test_get_pricing_exact_match(self):
    """Test getting pricing for an exact model name match."""
    service = GeminiPricingService()
    pricing = await service.get_pricing("gemini-2.5-pro")
    assert pricing is not None
    assert pricing.input_price_low == 1.25

  @pytest.mark.asyncio
  async def test_get_pricing_fuzzy_match(self):
    """Test getting pricing for a model with version suffix."""
    service = GeminiPricingService()
    pricing = await service.get_pricing("gemini-2.5-flash-001")
    assert pricing is not None
    assert pricing.input_price_low == 0.30

  @pytest.mark.asyncio
  async def test_get_pricing_with_prefix(self):
    """Test getting pricing for a model with 'models/' prefix."""
    service = GeminiPricingService()
    pricing = await service.get_pricing("models/gemini-2.0-flash")
    assert pricing is not None
    assert pricing.input_price_low == 0.15

  @pytest.mark.asyncio
  async def test_get_pricing_unknown_model(self):
    """Test getting pricing for an unknown model."""
    service = GeminiPricingService()
    pricing = await service.get_pricing("unknown-model-xyz")
    assert pricing is None


class TestCalculateTokenCost:
  """Test the calculate_token_cost helper function."""

  @pytest.mark.asyncio
  async def test_calculate_token_cost_gemini_25_pro(self):
    """Test cost calculation for Gemini 2.5 Pro."""
    cost = await calculate_token_cost("gemini-2.5-pro", 1000, 500, 0)
    assert cost is not None
    expected = (1000 / 1_000_000) * 1.25 + (500 / 1_000_000) * 10.00
    assert abs(cost - expected) < 0.000001

  @pytest.mark.asyncio
  async def test_calculate_token_cost_gemini_25_flash(self):
    """Test cost calculation for Gemini 2.5 Flash."""
    cost = await calculate_token_cost("gemini-2.5-flash", 10_000, 5_000, 0)
    assert cost is not None
    expected = (10_000 / 1_000_000) * 0.30 + (5_000 / 1_000_000) * 2.50
    assert abs(cost - expected) < 0.000001

  @pytest.mark.asyncio
  async def test_calculate_token_cost_with_cache(self):
    """Test cost calculation with cached tokens."""
    cost = await calculate_token_cost("gemini-2.5-pro", 1000, 500, 5000)
    assert cost is not None
    expected = (
        (1000 / 1_000_000) * 1.25
        + (5000 / 1_000_000) * 0.125
        + (500 / 1_000_000) * 10.00
    )
    assert abs(cost - expected) < 0.000001

  @pytest.mark.asyncio
  async def test_calculate_token_cost_unknown_model(self):
    """Test cost calculation for unknown model."""
    cost = await calculate_token_cost("unknown-model", 1000, 500, 0)
    assert cost is None
