---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/users
title: Stack Overflow Users
description: Contains information about registered users on the Stack Overflow platform.
tags: stackoverflow, users, community, reputation
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:51:02+00:00'
sources:
- resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/users
  title: 'BigQuery Table: users'
  id: bq-table-users
---

The `users` table in the [stackoverflow](../datasets/stackoverflow.md) dataset provides a comprehensive profile for each registered user on the Stack Overflow platform. Each row in this table represents a unique user, capturing details such as their display name, reputation score, activity dates, and biographical information. This table is essential for analyzing user behavior, community engagement, and overall platform dynamics.

# Schema

- `id`: Unique identifier for the user.
- `display_name`: The public display name chosen by the user.
- `about_me`: A short biography provided by the user.
- `age`: User's age (as a string, if provided).
- `creation_date`: Timestamp when the user account was created.
- `last_access_date`: Timestamp of the user's last activity or login.
- `location`: The geographical location provided by the user.
- `reputation`: The user's reputation score.
- `up_votes`: Total number of upvotes received by the user.
- `down_votes`: Total number of downvotes received by the user.
- `views`: Number of times the user's profile has been viewed.
- `profile_image_url`: URL to the user's profile picture.
- `website_url`: URL to the user's personal website.

# Common query patterns

```sql
-- Get the top 10 users by reputation
SELECT
    display_name,
    reputation,
    location
FROM
    `bigquery-public-data.stackoverflow.users`
ORDER BY
    reputation DESC
LIMIT 10;
```

```sql
-- Find users who joined in 2020 and have a high number of upvotes
SELECT
    id,
    display_name,
    creation_date,
    up_votes
FROM
    `bigquery-public-data.stackoverflow.users`
WHERE
    EXTRACT(YEAR FROM creation_date) = 2020
    AND up_votes > 1000
ORDER BY
    up_votes DESC
LIMIT 5;
```

```sql
-- Count users by location (top 5 locations)
SELECT
    location,
    COUNT(id) AS user_count
FROM
    `bigquery-public-data.stackoverflow.users`
WHERE
    location IS NOT NULL AND location != ''
GROUP BY
    location
ORDER BY
    user_count DESC
LIMIT 5;
```
