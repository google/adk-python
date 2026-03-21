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

"""Mock x402 server for testing the payment agent.

Run this before starting the agent::

    python contributing/samples/x402_payment_agent/mock_server.py

Endpoints:

*   ``GET /v1/market-data`` — returns HTTP 402 with x402 payment
    requirements unless a valid ``X-PAYMENT`` header is present.
"""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler
from http.server import HTTPServer
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

PORT = 8402

MOCK_MARKET_DATA = {
    "symbol": "NVDA",
    "price": 142.50,
    "volume": 45_000_000,
    "change_pct": 3.2,
    "timestamp": "2026-03-21T00:00:00Z",
    "source": "mock-x402-server",
}

PAYMENT_REQUIREMENTS = {
    "amount": "0.05",
    "payTo": "0x742d35Cc6634C0532925a3b844Bc9e7595f2bD18",
    "asset": "USDC",
    "network": "base",
    "description": "Premium market data — single query",
}


class X402Handler(BaseHTTPRequestHandler):
  """Handle x402 payment negotiation."""

  def do_GET(self):
    if self.path != "/v1/market-data":
      self._respond(404, {"error": "Not found"})
      return

    payment_header = self.headers.get("X-PAYMENT")

    if not payment_header:
      # Return 402 with payment requirements
      logger.info("No payment header — returning 402")
      self._respond(402, PAYMENT_REQUIREMENTS)
      return

    # Validate payment header (mock: accept anything parseable)
    try:
      payment = json.loads(payment_header)
      if "proof" not in payment:
        self._respond(400, {"error": "Missing payment proof"})
        return
    except json.JSONDecodeError:
      self._respond(400, {"error": "Invalid payment header"})
      return

    logger.info(
        "Payment accepted: %s %s",
        payment.get("amount", "?"),
        payment.get("asset", "?"),
    )
    self._respond(200, MOCK_MARKET_DATA)

  def _respond(self, status: int, data: dict) -> None:
    self.send_response(status)
    self.send_header("Content-Type", "application/json")
    self.end_headers()
    self.wfile.write(json.dumps(data, indent=2).encode())

  def log_message(self, format, *args):
    logger.debug(format, *args)


def main():
  server = HTTPServer(("0.0.0.0", PORT), X402Handler)
  logger.info("Mock x402 server on http://localhost:%d", PORT)
  logger.info("  GET /v1/market-data → 402 (no payment) or 200 (with payment)")
  try:
    server.serve_forever()
  except KeyboardInterrupt:
    server.shutdown()


if __name__ == "__main__":
  main()
