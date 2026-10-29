---
type: Reference
resource: https://support.google.com/analytics/answer/9037342
title: Highly Active Users Metric
description: Builds an audience of users active for more than N minutes in the last
  M days.
tags:
- metric
- audience
- ga4
- high-actives
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:16:29+00:00'
sources:
- title: Sample queries for audiences based on BigQuery data - Analytics Help
  resource: https://support.google.com/analytics/answer/9037342
  id: sample_queries
---

Builds an audience of Highly Active Users, defined as users who have been active/engaged for more than N minutes in the last M days, where M > N (for example, more than 0.1 minutes in the last 10 days).

# Schema
This reference describes a query pattern and does not map to a single database schema.

# Common query patterns

```sql
/**
 * Builds an audience of Highly Active Users.
 *
 * Highly Active Users = users who have been active for more than N minutes
 * in the last M days where M > N.
*/

SELECT
  COUNT(DISTINCT user_id) AS high_active_users_count
FROM
  (
    SELECT
      user_id,
      event_params.key,
      SUM(event_params.value.int_value)
    FROM
      `YOUR_TABLE.events_*` AS T
    CROSS JOIN
      T.event_params
    WHERE
      -- User engagement in the last M = 10 days.
      event_timestamp >
          UNIX_MICROS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 10 DAY))
      AND event_params.key = 'engagement_time_msec'
      AND _TABLE_SUFFIX BETWEEN '20180521' AND '20240131'
    GROUP BY 1, 2
    HAVING
      -- Having engaged for more than N = 0.1 minutes.
      SUM(event_params.value.int_value) > 0.1 * 60 * 1000000
  );
```
[^sample_queries]

[^sample_queries]: [Google Analytics Help: Sample queries for audiences based on BigQuery data](https://support.google.com/analytics/answer/9037342)
