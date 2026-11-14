---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_moderator_nomination
title: Posts Moderator Nomination
description: Contains posts related to moderator nominations on the Stack Overflow
  platform.
tags: stackoverflow, posts, moderator, nomination
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:48:22+00:00'
sources:
- title: 'BigQuery Table: posts_moderator_nomination'
  id: posts-moderator-nomination-table
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_moderator_nomination
---

This table stores posts that represent moderator nominations within the Stack Overflow community. Each row corresponds to a single nomination post, typically detailing why an individual should be considered for a moderator role. These posts are characterized by `post_type_id = 6`. The table includes information such as the post's content (`body`), creation date, and the user who owns the post.

# Schema

- `id` (INTEGER): Unique identifier for the post.
- `title` (STRING): The title of the post.
- `body` (STRING): The main content of the moderator nomination post.
- `accepted_answer_id` (STRING)
- `answer_count` (STRING)
- `comment_count` (INTEGER): The number of comments on the post.
- `community_owned_date` (TIMESTAMP)
- `creation_date` (TIMESTAMP): The date and time the post was created.
- `favorite_count` (STRING)
- `last_activity_date` (TIMESTAMP)
- `last_edit_date` (TIMESTAMP)
- `last_editor_display_name` (STRING)
- `last_editor_user_id` (INTEGER)
- `owner_display_name` (STRING)
- `owner_user_id` (INTEGER): The ID of the user who owns the post.
- `parent_id` (STRING)
- `post_type_id` (INTEGER): Indicates the type of post; `6` for moderator nominations.
- `score` (INTEGER): The score of the post.
- `tags` (STRING): Tags associated with the post.
- `view_count` (STRING)

# Common query patterns

```sql
SELECT
  id,
  creation_date,
  owner_user_id,
  body
FROM
  `bigquery-public-data.stackoverflow.posts_moderator_nomination`
WHERE
  creation_date >= '2020-01-01'
LIMIT 100;
```
