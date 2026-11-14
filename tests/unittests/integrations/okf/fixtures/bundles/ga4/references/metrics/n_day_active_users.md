---
type: Reference
resource: https://support.google.com/analytics/answer/9037342
title: N-Day Active Users Metric
description: Builds an audience of users active in the last N days based on engagement_time_msec.
tags:
- metric
- audience
- ga4
- active-users
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:16:16+00:00'
sources:
- title: Sample queries for audiences based on BigQuery data - Analytics Help
  resource: https://support.google.com/analytics/answer/9037342
  id: sample_queries
---

Builds an audience of N-Day Active Users, defined as users who have logged at least one event with the event parameter `engagement_time_msec > 0` in the last N days.

# Schema
This reference describes a query pattern and does not map to a single database schema.

# Common query patterns

```sql
/**
 * Builds an audience of N-Day Active Users.
 *
 * N-day active users = users who have logged at least one event with event param
 * engagement_time_msec > 0 in the last N days.
*/

SELECT
  COUNT(DISTINCT user_id) AS n_day_active_users_count
FROM
  `YOUR_TABLE.events_*` AS T
    CROSS JOIN
      T.event_params
WHERE
  event_params.key = 'engagement_time_msec' AND event_params.value.int_value > 0
  -- Pick events in the last N = 20 days.
  AND event_timestamp >
      UNIX_MICROS(TIMESTAMP_SUB(CURRENT_TIMESTAMP, INTERVAL 20 DAY))
  AND _TABLE_SUFFIX BETWEEN '20180521' AND '20240131';
```
[^sample_queries]

[^sample_queries]: [Google Analytics Help: Sample queries for audiences based on BigQuery data](https://support.google.com/analytics/answer/9037342)
