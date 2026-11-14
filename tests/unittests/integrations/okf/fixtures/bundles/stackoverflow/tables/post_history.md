---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/post_history
title: Post History
description: Records the history of all changes and events related to posts on Stack
  Overflow.
tags:
- stackoverflow
- posts
- history
- changes
- events
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:47:32+00:00'
sources:
- id: post-history-resource
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/post_history
  title: Post History BigQuery Table
---

The `post_history` table in the [stackoverflow](../datasets/stackoverflow.md) dataset records a detailed history of all changes and events associated with posts on Stack Overflow. Each row represents a specific historical event or revision made to a post, including initial creation, edits, rollbacks, and changes in status. This table is crucial for auditing post evolution and understanding how content changes over time. Records date back to October 2016, with a total of over 150 million entries.

# Schema

- `id`: Unique identifier for each post history entry.
- `creation_date`: Timestamp when this history entry was created.
- `post_id`: The ID of the post to which this history entry belongs. This links to the `id` field in the [posts_questions](../tables/posts_questions.md) and [posts_answers](../tables/posts_answers.md) tables.
- `post_history_type_id`: An integer ID representing the type of history event (e.g., initial title, body edit, tag edit). Specific meanings for these IDs are typically found in a separate lookup table.
- `revision_guid`: A unique GUID used to group related history entries that constitute a single revision.
- `user_id`: The ID of the user who performed this action. This links to the `id` field in the [users](../tables/users.md) table.
- `text`: The content associated with this history entry. Its meaning depends on `post_history_type_id` (e.g., new title, new body content, old tags).
- `comment`: An optional comment provided by the user for this change.

# Common query patterns

**Retrieve all history entries for a specific post:**
```sql
SELECT
  id,
  creation_date,
  post_id,
  post_history_type_id,
  revision_guid,
  user_id,
  text,
  comment
FROM
  `bigquery-public-data.stackoverflow.post_history`
WHERE
  post_id = 12345 -- Replace with a specific post ID
ORDER BY
  creation_date DESC;
```

**Find the most recent edits made by a specific user:**
```sql
SELECT
  ph.creation_date,
  ph.post_id,
  ph.text,
  ph.comment,
  pq.title AS post_title
FROM
  `bigquery-public-data.stackoverflow.post_history` AS ph
JOIN
  `bigquery-public-data.stackoverflow.posts_questions` AS pq ON ph.post_id = pq.id
WHERE
  ph.user_id = 67890 -- Replace with an actual user ID
  AND ph.post_history_type_id IN (2, 5) -- Example: assuming 2 and 5 represent body/title edits
ORDER BY
  ph.creation_date DESC
LIMIT 10;
```

**Count distinct types of history events for a given post:**
```sql
SELECT
  post_history_type_id,
  COUNT(*) AS event_count
FROM
  `bigquery-public-data.stackoverflow.post_history`
WHERE
  post_id = 12345 -- Replace with a specific post ID
GROUP BY
  post_history_type_id
ORDER BY
  event_count DESC;
```
