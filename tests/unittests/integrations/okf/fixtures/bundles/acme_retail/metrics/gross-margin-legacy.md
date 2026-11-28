---
type: Metric
title: Gross Margin (legacy, pre-FY2026)
description: Retired gross-margin definition that included only product cost. Preserved for historical query reproducibility. Do not use for new analyses.
tags: [finance, margin, deprecated]
generated: { by: human:jsmith@acme, at: 2024-01-15T10:00:00Z }
verified:
  - { by: human:jsmith@acme, at: 2024-01-15T10:00:00Z }
status: deprecated
---

# Deprecated

**This metric is retired.** The current gross-margin definition is [`metrics/gross-margin.md`](./gross-margin.md), which implements the FY2026 Cost Allocation Standard (product cost + inbound fulfillment + outbound shipping + payment fees).

This concept is preserved so historical reports written before 2026-02-01 remain reproducible. Do not reference it for new work.

# Legacy definition (for reproducibility only)

Under the pre-FY2026 definition, gross margin was:

```
gross-margin-legacy(period) = revenue(period) - SUM(products.cost * order_lines.quantity)  over orders recognized in period
```

That is, COGS was product cost only; fulfillment, shipping, and payment fees were booked to operating expenses rather than COGS. Finance concluded in Q4 2025 that this understated the operational cost of goods and made the number unreconcilable to the general ledger.

# Why no attested computation

There is no `Attested Computation` for this metric. When it was retired, its SQL was deleted from the sanctioned set. Anyone re-running historical reports must reconstruct the SQL from this narrative and clearly label the result as legacy.
