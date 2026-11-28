---
type: Reference
resource: https://support.google.com/analytics/answer/9037342
title: Google Acquired Cohorts Metric
description: Builds an audience of users acquired in a specific time-window cohort
  filtered by Google campaign source.
tags:
- metric
- audience
- ga4
- cohorts
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T21:16:43+00:00'
sources:
- title: Sample queries for audiences based on BigQuery data - Analytics Help
  resource: https://support.google.com/analytics/answer/9037342
  id: sample_queries
---

Builds an audience composed of users acquired last week through Google campaigns (cohorts with specific campaign filters).

# Schema
This reference describes a query pattern and does not map to a single database schema.

# Common query patterns

```sql
/**
 * Builds an audience composed of users acquired last week
 * through Google campaigns, i.e., cohorts with filters.
 *
 * Cohort is defined as users acquired last week, i.e. between 7 - 14
 * days ago. The cohort filter is for users acquired through a direct
 * campaign.
 */

SELECT
  COUNT(DISTINCT user_id) AS users_acquired_through_google_count
FROM
  `YOUR_TABLE.events_*`
WHERE
  event_name = 'first_open'
  -- Cohort: opened app 1-2 weeks ago. One week of cohort, aka. weekly.
  AND event_timestamp >
      UNIX_MICROS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 14 DAY))
  AND event_timestamp <
      UNIX_MICROS(TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY))
  -- Cohort filter: users acquired through 'google' source.
  AND traffic_source.source = 'google'
  AND _TABLE_SUFFIX BETWEEN '20180501' AND '20240131';
```
[^sample_queries]

[^sample_queries]: [Google Analytics Help: Sample queries for audiences based on BigQuery data](https://support.google.com/analytics/answer/9037342)
