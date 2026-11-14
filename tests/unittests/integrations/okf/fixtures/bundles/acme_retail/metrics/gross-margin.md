---
type: Metric
title: Gross Margin
description: Gross margin for a period, per Acme's FY2026 Cost Allocation Standard (product cost + inbound fulfillment + outbound shipping + payment fees).
tags: [finance, margin, headline-metric]
generated: { by: reference_agent/gemini-2.5-pro, at: 2026-06-30T14:00:00Z }
verified:
  - { by: human:jsmith@acme, at: 2026-07-01T09:00:00Z }
status: stable
stale_after: 2026-12-31T00:00:00Z
not:
  - term: "revenue minus product cost only"
    why: "that is the pre-FY2026 definition (see gross-margin-legacy). It excluded fulfillment, shipping, and payment fees, and could not be reconciled to the general ledger."
    instead: "revenue minus full COGS (product cost + inbound fulfillment + outbound shipping + payment fees)"
sources:
  - id: margin-standard
    resource: policies/margin-standard.md
    title: Cost Allocation & Margin Standard (FY2026)
    author: human:jsmith@acme
    last_modified: 2026-06-15T00:00:00Z
  - id: revenue-policy
    resource: policies/revenue-recognition.md
    title: Revenue Recognition Policy (FY2026)
    author: human:jsmith@acme
    last_modified: 2026-06-15T00:00:00Z
---

# Definition

**Not:** revenue minus product cost only (that was the pre-2026 formula; see [`gross-margin-legacy`](./gross-margin-legacy.md)).

Gross margin for a period equals recognized [Revenue](./revenue.md) minus **full COGS**, where full COGS is the sum of product cost, inbound fulfillment cost, outbound shipping cost, and payment processing fees. [^margin-standard]

```
gross_margin(period) = revenue(period) - cogs_full(period)
```

The sanctioned computation is [`computations/gross-margin-period.md`](../computations/gross-margin-period.md). Consumers MUST run and attest that computation.

# What changed in FY2026

Prior to 2026-02-01, Acme's gross-margin definition included only product cost, excluding fulfillment, shipping, and payment fees. That legacy definition is preserved in [`metrics/gross-margin-legacy.md`](./gross-margin-legacy.md) as `status: deprecated` for historical query reproducibility.

The switch reduced reported gross margin by roughly 4-6 percentage points depending on category. It also brought the number in line with the general ledger, closing a long-standing reconciliation gap.

# Trust and freshness

- **Verified:** VP Finance sign-off on 2026-07-01, against the FY2026 margin standard.
- **Stale after 2026-12-31:** the cost-allocation standard is reviewed annually. Consumers must re-verify against the FY2027 standard before serving.

[^margin-standard]: Cost Allocation & Margin Standard (FY2026)
[^revenue-policy]: Revenue Recognition Policy (FY2026)
