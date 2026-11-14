---
type: Metric
title: Revenue
description: Recognized revenue for a period, per Acme's FY2026 revenue-recognition policy. Backed by an Attested Computation.
tags: [finance, revenue, headline-metric]
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
---

# Definition

Revenue for a fiscal year is the sum of `net_amount` over orders that (a) reached `order_status = 'delivered'`, (b) completed the 30-day return window, and (c) fall in the fiscal year by `order_ts`. Multi-currency orders are converted to USD at the `order_ts` daily reference rate. [^revenue-policy]

The sanctioned computation is [`computations/revenue-ytd.md`](../computations/revenue-ytd.md). Consumers MUST run and attest that computation rather than composing their own SUM. The attester rejects any receipt whose executed SQL does not match the sanctioned form.

# Reporting cuts

- **By fiscal year:** the sanctioned computation takes `year` as its sole parameter.
- **By channel or category:** these are approved narrations, not new metrics. Join the receipt's row-level result to `orders.channel` or to `order_lines` × `products.category` client-side. Do NOT rewrite the sanctioned SQL.

# Trust and freshness

- **Verified:** VP Finance sign-off on 2026-07-01, against the FY2026 policy.
- **Stale after 2026-12-31:** Finance re-issues the revenue recognition policy each January. Consumers of this concept after 2027-01-01 MUST re-verify the definition against the new policy before serving.

[^revenue-policy]: Revenue Recognition Policy (FY2026)
