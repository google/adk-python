---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_privilege_wiki
title: Stack Overflow Privilege Wiki Posts
description: Contains information about Stack Overflow's privilege wiki posts, detailing
  user capabilities and their requirements.
tags:
- stackoverflow
- wiki
- privilege
- posts
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:49:00+00:00'
sources:
- resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_privilege_wiki
  id: posts-privilege-wiki-resource
  title: Stack Overflow posts_privilege_wiki Table
---

This table, part of the [Stack Overflow dataset](../datasets/stackoverflow.md), contains specific posts from the Stack Overflow platform that describe various user privileges. These "privilege wiki" posts detail the capabilities users gain at different reputation levels, such as the ability to edit questions and answers or retag questions without peer review. Each entry in this table represents a single privilege explanation, providing a comprehensive description of the privilege and its implications.

The `posts_privilege_wiki` table is characterized by `post_type_id` equal to `8`, indicating its nature as a wiki post specifically for privileges. It allows users to understand the mechanics of reputation and privileges on the platform.

# Schema

- id: INTEGER
- title: STRING
- body: STRING
    The full HTML content of the privilege wiki post, describing the privilege in detail.
- accepted_answer_id: STRING
- answer_count: STRING
- comment_count: INTEGER
- community_owned_date: STRING
- creation_date: TIMESTAMP
    The date and time when the privilege wiki post was originally created.
- favorite_count: STRING
- last_activity_date: TIMESTAMP
- last_edit_date: TIMESTAMP
    The date and time when the privilege wiki post was last edited.
- last_editor_display_name: STRING
- last_editor_user_id: INTEGER
- owner_display_name: STRING
- owner_user_id: INTEGER
    The ID of the user who owns or created the privilege wiki post.
- parent_id: STRING
- post_type_id: INTEGER
    Always `8` for privilege wiki posts.
- score: INTEGER
- tags: STRING
- view_count: STRING

# Common query patterns

```sql
-- Retrieve all privilege wiki posts
SELECT
    id,
    title,
    body,
    creation_date
  FROM
    `bigquery-public-data.stackoverflow.posts_privilege_wiki`
  WHERE
    post_type_id = 8
  LIMIT 100;
```

```sql
-- Find privilege wiki posts mentioning "edit" in their body
SELECT
    id,
    title,
    creation_date
  FROM
    `bigquery-public-data.stackoverflow.posts_privilege_wiki`
  WHERE
    post_type_id = 8 AND CONTAINS_SUBSTR(body, 'edit');
```

```sql
-- Count the number of privilege wiki posts
SELECT
    COUNT(id) AS privilege_wiki_post_count
  FROM
    `bigquery-public-data.stackoverflow.posts_privilege_wiki`
  WHERE
    post_type_id = 8;
```
