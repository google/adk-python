---
type: Reference
resource: https://support.google.com/analytics/answer/9037342
title: N-Day Inactive Users Metric
description: Builds an audience of users active in the last M days who have not been
  active in the last N days.
tags:
- metric
- audience
- ga4
- inactive-users
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:16:21+00:00'
sources:
- title: Sample queries for audiences based on BigQuery data - Analytics Help
  id: sample_queries
  resource: https://support.google.com/analytics/answer/9037342
---

Builds an audience of N-Day Inactive Users. Inactive users are defined as those active in the last M days (e.g. 7 days) who have NOT logged any event with event parameter `engagement_time_msec > 0` in the last N days (e.g. 2 days), where M > N.

# Schema
This reference describes a query pattern and does not map to a single database schema.

# Common query patterns

```sql
/**
 * Builds an audience of N-Day Inactive Users.
 *
 * N-Day inactive users = users in the last M days who have not logged one
 * event with event param engagement_time_msec > 0 in the last N days
 *  where M > N.
 */

SELECT
  COUNT(DISTINCT MDaysUsers.user_id) AS n_day_inactive_users_count
FROM
  (
    SELECT
      user_id
    FROM
      `YOUR_TABLE.events_*` AS T
    CROSS JOIN
      T.event_params
    WHERE
      event_params.key = 'engagement_time_msec' AND event_params.value.int_value > 0
      /* Has engaged in last M = 7 days */
      AND event_timestamp >
          UNIX_MICROS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY))
      AND _TABLE_SUFFIX BETWEEN '20180521' AND '20240131'
  ) AS MDaysUsers
LEFT JOIN
  (
    SELECT
      user_id
    FROM
      `YOUR_TABLE.events_*` AS T
    CROSS JOIN
      T.event_params
    WHERE
      event_params.key = 'engagement_time_msec' AND event_params.value.int_value > 0
      /* Has engaged in last N = 2 days */
      AND event_timestamp >
          UNIX_MICROS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 2 DAY))
      AND _TABLE_SUFFIX BETWEEN '20180521' AND '20240131'
  ) AS NDaysUsers
  ON MDaysUsers.user_id = NDaysUsers.user_id
WHERE
  NDaysUsers.user_id IS NULL;
```
[^sample_queries]

[^sample_queries]: [Google Analytics Help: Sample queries for audiences based on BigQuery data](https://support.google.com/analytics/answer/9037342)
