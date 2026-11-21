---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/badges
title: Badges
description: This table contains information about badges awarded to users on Stack
  Overflow.
tags:
- stackoverflow
- badges
- gamification
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:59:00+00:00'
sources:
- resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/badges
  title: Stack Overflow Badges Table
  id: badges-table
---

The `badges` table tracks all badges awarded to users on the Stack Overflow platform. Each row represents a single badge instance awarded to a specific user at a specific time. The table includes details such as the badge's name, the user who received it, and whether it's a tag-based badge. This data can be used to analyze user engagement, recognize top contributors, and understand the gamification aspects of the platform.

# Schema
- `id`: INTEGER, Unique identifier for the badge award.
- `name`: STRING, Name of the awarded badge (e.g., "Great Answer", "Electorate").
- `date`: TIMESTAMP, The date and time the badge was awarded.
- `user_id`: INTEGER, The ID of the user who received the badge. Links to the [users](../tables/users.md) table.
- `class`: INTEGER, The class or tier of the badge (e.g., 1 for gold, 2 for silver, 3 for bronze).
- `tag_based`: BOOLEAN, Indicates whether the badge is associated with a specific tag.

# Common query patterns
1. **Count the number of badges awarded per user:**
   ```sql
   SELECT
     user_id,
     count(id) AS badge_count
   FROM
     `bigquery-public-data.stackoverflow.badges`
   GROUP BY
     user_id
   ORDER BY
     badge_count DESC
   LIMIT 10;
   ```
2. **Find the most frequently awarded badges:**
   ```sql
   SELECT
     name,
     count(id) AS award_count
   FROM
     `bigquery-public-data.stackoverflow.badges`
   GROUP BY
     name
   ORDER BY
     award_count DESC
   LIMIT 10;
   ```
3. **Get all gold badges awarded to a specific user:**
   ```sql
   SELECT
     t2.display_name,
     t1.name,
     t1.date
   FROM
     `bigquery-public-data.stackoverflow.badges` AS t1
   JOIN
     `bigquery-public-data.stackoverflow.users` AS t2
   ON
     t1.user_id = t2.id
   WHERE
     t1.class = 1 -- Assuming class 1 is Gold
     AND t2.display_name = 'Jon Skeet'
   ORDER BY
     t1.date DESC;
   ```
