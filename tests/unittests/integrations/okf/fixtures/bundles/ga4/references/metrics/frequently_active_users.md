---
type: Reference
resource: https://support.google.com/analytics/answer/9037342
title: Frequently Active Users Metric
description: Builds an audience of users active on at least N of the last M days.
tags:
- metric
- audience
- ga4
- frequent-actives
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:16:25+00:00'
sources:
- title: Sample queries for audiences based on BigQuery data - Analytics Help
  id: sample_queries
  resource: https://support.google.com/analytics/answer/9037342
---

Builds an audience of Frequently Active Users, defined as users who have logged at least one event with the event parameter `engagement_time_msec > 0` on N of the last M days, where M > N (for example, on at least 4 of the last 10 days).

# Schema
This reference describes a query pattern and does not map to a single database schema.

# Common query patterns

```sql
/**
 * Builds an audience of Frequently Active Users.
 *
 * Frequently Active Users = users who have logged at least one
 * event with event param engagement_time_msec > 0 on N of
 * the last M days where M > N.
 */

SELECT
  COUNT(DISTINCT user_id) AS frequent_active_users_count
FROM
  (
    SELECT
      user_id,
      COUNT(DISTINCT event_date)
    FROM
      `YOUR_TABLE.events_*` AS T
    CROSS JOIN
      T.event_params
    WHERE
      event_params.key = 'engagement_time_msec' AND event_params.value.int_value > 0
      -- User engagement in the last M = 10 days.
      AND event_timestamp >
          UNIX_MICROS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 10 DAY))
      AND _TABLE_SUFFIX BETWEEN '20180521' AND '20240131'
    GROUP BY 1
    -- Having engaged in at least N = 4 days.
    HAVING COUNT(event_date) >= 4
  );
```
[^sample_queries]

[^sample_queries]: [Google Analytics Help: Sample queries for audiences based on BigQuery data](https://support.google.com/analytics/answer/9037342)
