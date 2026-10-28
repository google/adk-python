---
type: Reference
resource: https://support.google.com/analytics/answer/9037342
title: Acquired Users Metric
description: Builds an audience of users acquired via a specific Source, Medium, and
  Campaign name.
tags:
- metric
- audience
- ga4
- acquired-users
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:16:35+00:00'
sources:
- title: Sample queries for audiences based on BigQuery data - Analytics Help
  resource: https://support.google.com/analytics/answer/9037342
  id: sample_queries
---

Builds an audience of Acquired Users, defined as users who were acquired via a specific marketing campaign source, medium, and name.

# Schema
This reference describes a query pattern and does not map to a single database schema.

# Common query patterns

```sql
/**
 * Builds an audience of Acquired Users.
 *
 * Acquired Users = users who were acquired via some Source/Medium/Campaign.
 */

SELECT
  COUNT(DISTINCT user_id) AS acquired_users_count
FROM
  `YOUR_TABLE.events_*`
WHERE
  traffic_source.source = 'google'
  AND traffic_source.medium = 'cpc'
  AND traffic_source.name = 'VTA-Test-Android'
  AND _TABLE_SUFFIX BETWEEN '20180521' AND '20240131';
```
[^sample_queries]

[^sample_queries]: [Google Analytics Help: Sample queries for audiences based on BigQuery data](https://support.google.com/analytics/answer/9037342)
