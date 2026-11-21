---
type: BigQuery Table
title: Customer Orders
description: One row per completed customer order across web, mobile, and marketplace channels. The grain is the order, not the line item; per-line product detail lives in `order_lines`.
resource: https://console.cloud.google.com/bigquery?p=acme&d=sales&t=orders
tags: [sales, orders, revenue]
generated: { by: reference_agent/gemini-2.5-pro, at: 2026-06-30T14:00:00Z }
verified:
  - { by: human:kliu@acme, at: 2026-07-01T16:00:00Z }
status: stable
stale_after: 2026-12-31T00:00:00Z
sources:
  - id: warehouse-schema
    resource: https://wiki.acme.internal/data/warehouse/schemas/sales
    title: Acme Retail warehouse schema — sales dataset
    author: team:data-platform
    usage_count: 1240
    last_modified: 2026-06-15T00:00:00Z
  - id: revenue-policy
    resource: policies/revenue-recognition.md
    title: Revenue Recognition Policy (FY2026)
    author: human:jsmith@acme
    last_modified: 2026-06-15T00:00:00Z
usage_window: { from: 2026-04-01T00:00:00Z, to: 2026-06-30T00:00:00Z }
---

# Schema

| Column | Type | Description |
|---|---|---|
| `order_id` | STRING | Globally unique order id. Generated at order creation. [^warehouse-schema] |
| `customer_id` | STRING | FK into `customers`. Never null for completed orders. [^warehouse-schema] |
| `order_ts` | TIMESTAMP | Order placement time in UTC. This is the timestamp used for fiscal-year assignment. [^revenue-policy] |
| `order_status` | STRING | One of `pending`, `paid`, `shipped`, `delivered`, `cancelled`, `refunded`. Revenue is recognized only when `order_status = 'delivered'` and the 30-day return window has closed. [^revenue-policy] |
| `gross_amount` | NUMERIC(18,4) | Pre-discount subtotal, in `currency`. Excludes tax and shipping. [^warehouse-schema] |
| `discount_amount` | NUMERIC(18,4) | Total discounts applied (promo codes, loyalty credits, price adjustments). [^warehouse-schema] |
| `net_amount` | NUMERIC(18,4) | `gross_amount - discount_amount`. This is the recognized-revenue amount per policy. [^revenue-policy] |
| `shipping_amount` | NUMERIC(18,4) | Carrier charge billed to the customer. Excluded from revenue (pass-through liability). [^revenue-policy] |
| `tax_amount` | NUMERIC(18,4) | Sales tax collected. Excluded from revenue (pass-through liability). [^revenue-policy] |
| `currency` | STRING | ISO 4217 currency code. Non-USD orders convert via `finance.fx_daily_rates` on `order_ts` date. [^revenue-policy] |
| `channel` | STRING | Order origin: `web`, `mobile`, `marketplace`. Marketplace orders (Amazon, eBay) are net-settled and recognized on marketplace payout, not on `order_status = 'delivered'`. |

# Notes for consumers

- The grain assumption trips up new analysts: `SUM(net_amount) GROUP BY order_id` is a no-op because there is exactly one row per order. For per-SKU revenue, join `order_lines`.
- The `refunded` status is terminal in this table; the refund event itself lives in `finance.refunds`, keyed on `order_id`.

[^warehouse-schema]: Acme Retail warehouse schema — sales dataset
[^revenue-policy]: Revenue Recognition Policy (FY2026)
