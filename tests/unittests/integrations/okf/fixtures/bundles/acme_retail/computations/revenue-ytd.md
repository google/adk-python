---
type: Attested Computation
title: Revenue for a fiscal year
description: Sanctioned SQL that produces the recognized-revenue figure for a given fiscal year, per Acme's FY2026 Revenue Recognition Policy.
tags: [finance, revenue, attested]
runtime: bigquery
parameters:
  - { name: year, type: integer, required: true }
executor:
  resource: skills/run-on-bq.md
  receipt: [job_id, executed_sql, result]
attester:
  resource: attesters/sql_equality.py
generated: { by: reference_agent/gemini-2.5-pro, at: 2026-06-30T14:00:00Z }
verified:
  - { by: human:jsmith@acme, at: 2026-07-01T09:00:00Z }
status: stable
stale_after: 2026-12-31T00:00:00Z
sources:
  - id: revenue-policy
    resource: policies/revenue-recognition.md
    title: Revenue Recognition Policy (FY2026)
    author: human:jsmith@acme
    last_modified: 2026-06-15T00:00:00Z
  - id: orders-table
    resource: tables/orders.md
    title: Customer Orders (BigQuery table)
    author: team:data-platform
    last_modified: 2026-07-01T00:00:00Z
---

# Computation

```sql
SELECT
  SUM(
    CASE
      WHEN o.currency = 'USD' THEN o.net_amount
      ELSE o.net_amount * fx.rate_to_usd
    END
  ) AS revenue_usd
FROM `acme.sales.orders` AS o
LEFT JOIN `acme.finance.fx_daily_rates` AS fx
  ON fx.currency = o.currency
  AND fx.rate_date = DATE(o.order_ts)
WHERE o.order_status = 'delivered'
  AND DATE_DIFF(CURRENT_DATE(), DATE(o.order_ts), DAY) >= 30
  AND EXTRACT(YEAR FROM o.order_ts) = @year
```

This computation implements the four rules of the FY2026 Revenue Recognition Policy: [^revenue-policy]

1. **Recognition trigger:** `order_status = 'delivered'` AND the 30-day return window has closed.
2. **Recognized amount:** `net_amount` (excludes shipping and tax).
3. **Currency:** non-USD orders convert at the `order_ts` daily rate.
4. **Fiscal year:** calendar year from `order_ts`.

# What the attester checks

`attesters/sql_equality.py` receives the receipt returned by `skills/run-on-bq.md` and verifies two things:

1. **Provenance:** `receipt.executed_sql`, canonicalized (whitespace, comment stripping, keyword casing), equals the SQL above canonicalized the same way. Any rewrite (a swapped table, an added filter, a dropped JOIN) fails the check.
2. **Fidelity:** the value the caller is about to display equals `receipt.result[0]`.

A run whose SQL does not match is treated as unattested; the consumer MUST refuse to display the value.

# Freshness

`stale_after: 2026-12-31T00:00:00Z` mirrors the revenue-recognition policy's annual review cycle. On 2027-01-01, a consumer running this computation SHOULD flag the result for re-verification before serving it, per the memory-aware consumer contract.

[^revenue-policy]: Revenue Recognition Policy (FY2026)
