---
type: Policy
title: Acme Retail — Revenue Recognition Policy (FY2026)
description: Finance policy defining when a customer order is recognized as revenue. Reviewed annually.
resource: https://wiki.acme.internal/finance/revenue-recognition
tags: [finance, policy, revenue]
generated: { by: human:jsmith@acme, at: 2026-01-05T10:00:00Z }
verified:
  - { by: human:jsmith@acme, at: 2026-06-15T09:00:00Z }
status: stable
stale_after: 2026-12-31T00:00:00Z
---

# Acme Retail Revenue Recognition Policy — FY2026

**Owner:** VP Finance (jsmith@acme)
**Effective:** 2026-01-01
**Next scheduled review:** 2026-12-31

## Recognition trigger

Revenue is recognized when a customer order reaches `order_status = 'delivered'` **and** the return window has closed (delivered date + 30 days). Orders in earlier statuses are backlog, not revenue.

## Recognized amount

The recognized amount for an order equals `net_amount = gross_amount - discount_amount`. Shipping and tax are excluded from revenue per US GAAP (they are pass-through liabilities).

## Currency

Multi-currency orders are converted at the daily reference rate published in `finance.fx_daily_rates`, using the `order_ts` date. All reporting is in USD.

## Refunds and cancellations

Refunds and post-recognition cancellations are booked as contra-revenue in the period of the refund, not by retroactive adjustment to the original recognition period.

## Fiscal year

Acme Retail operates on the US calendar year (Jan 1 – Dec 31). All fiscal-year metrics use `fiscal_year = EXTRACT(YEAR FROM order_ts)`.

## What this policy authorizes

Any Attested Computation whose `sources` cites this policy MUST implement the four rules above. Deviations require a policy addendum reviewed by Finance.

# Cited by

- [`tables/orders`](/tables/orders.md) — `order_status`, `order_ts`, and `net_amount` columns implement the recognition rules
- [`metrics/revenue`](/metrics/revenue.md) — the recognized-revenue definition derives from this policy
- [`metrics/gross-margin`](/metrics/gross-margin.md) — the revenue side of gross margin follows this policy
- [`computations/revenue-ytd`](/computations/revenue-ytd.md) — the sanctioned SQL implements all four rules
- [`computations/gross-margin-period`](/computations/gross-margin-period.md) — revenue leg uses these recognition rules
