---
type: Reference
resource: https://support.google.com/analytics/answer/9037342
title: Purchasers Audience Metric
description: Computes the count or list of users who have completed a purchase or
  in-app purchase.
tags:
- metric
- audience
- ga4
- purchasers
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:16:12+00:00'
sources:
- id: sample_queries
  title: Sample queries for audiences based on BigQuery data - Analytics Help
  resource: https://support.google.com/analytics/answer/9037342
---

Computes the audience of purchasers, defined as users who have logged either `in_app_purchase` or `purchase`.

# Schema
This reference describes a query pattern and does not map to a single database schema.

# Common query patterns

```sql
/**
 * Computes the audience of purchasers.
 *
 * Purchasers = users who have logged either in_app_purchase or
 * purchase.
 */

SELECT
  COUNT(DISTINCT user_id) AS purchasers_count
FROM
  `YOUR_TABLE.events_*`
WHERE
  event_name IN ('in_app_purchase', 'purchase')
  AND _TABLE_SUFFIX BETWEEN '20180501' AND '20240131';
```
[^sample_queries]

[^sample_queries]: [Google Analytics Help: Sample queries for audiences based on BigQuery data](https://support.google.com/analytics/answer/9037342)
