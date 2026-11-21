---
type: Policy
title: Acme Retail — Cost Allocation & Margin Standard (FY2026)
description: Finance policy defining COGS composition and the standard gross-margin formula. Introduced FY2026 (superseded a legacy definition that excluded fulfillment/shipping).
resource: https://wiki.acme.internal/finance/margin-standard
tags: [finance, policy, margin, cogs]
generated: { by: human:jsmith@acme, at: 2026-02-01T10:00:00Z }
verified:
  - { by: human:jsmith@acme, at: 2026-06-15T09:00:00Z }
status: stable
stale_after: 2026-12-31T00:00:00Z
---

# Acme Retail Cost Allocation & Margin Standard — FY2026

**Owner:** VP Finance (jsmith@acme)
**Effective:** 2026-02-01
**Supersedes:** the pre-2026 margin definition (see `metrics/gross-margin-legacy.md`)

## COGS composition

FY2026 COGS for a completed order is the sum of:

1. **Product cost** — from `products.cost` at time of order (locked at order creation)
2. **Inbound fulfillment cost** — allocated per-unit from monthly warehouse aggregates
3. **Outbound shipping cost** — carrier-billed actuals from `logistics.shipment_cost`
4. **Payment processing fees** — Stripe fees from `finance.payment_fees`

The pre-2026 definition included only (1). Adding (2) — (4) reduces reported gross margin by ~4-6 percentage points but produces a number Finance can reconcile to the GL.

## Gross margin formula

For a period P:

```
gross_margin(P) = SUM(net_amount) - SUM(cogs_full)   over orders recognized in P
```

Where `cogs_full` is the sum of the four components above.

## Reporting granularity

The standard supports gross margin at three levels: portfolio, category, and SKU. Category and SKU cuts require joining `orders` to `products` on `product_id`.

## What this policy authorizes

Any Attested Computation whose `sources` cites this policy MUST use all four COGS components. The legacy formula (product-cost-only) is preserved in `metrics/gross-margin-legacy.md` for historical query reproducibility; do not use it for new analyses.

# Cited by

- [`metrics/gross-margin`](/metrics/gross-margin.md) — current gross-margin metric implements this standard
- [`metrics/gross-margin-legacy`](/metrics/gross-margin-legacy.md) — deprecated metric, superseded by the definition this standard authorizes
- [`computations/gross-margin-period`](/computations/gross-margin-period.md) — sanctioned SQL implements the four COGS components defined here
