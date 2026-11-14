---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_tag_wiki
title: Posts Tag Wiki
description: Detailed wiki entries associated with tags used on Stack Overflow, providing
  comprehensive information beyond the basic tag descriptions.
tags:
- stackoverflow
- posts
- tags
- wiki
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:49:37+00:00'
sources:
- resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_tag_wiki
  title: 'BigQuery Table: posts_tag_wiki'
  id: bq-table
---

The `posts_tag_wiki` table in the [stackoverflow](../datasets/stackoverflow.md) dataset contains comprehensive wiki entries for tags used across the Stack Overflow platform. Unlike the brief descriptions found in the [tags](tags.md) table or the [posts_tag_wiki_excerpt](posts_tag_wiki_excerpt.md), this table provides full-length content for each tag's wiki, often including extensive explanations, usage guidelines, and examples. Each row represents a single tag wiki entry, identified by its unique `id`.

# Schema

- `id`: Unique identifier for the tag wiki entry.
- `title`: The title of the wiki entry (often NULL in this table, implying the tag itself is the title).
- `body`: The main content of the tag wiki, typically in HTML format.
- `accepted_answer_id`: ID of the accepted answer (if applicable).
- `answer_count`: Number of answers.
- `comment_count`: Number of comments.
- `community_owned_date`: Date when the post became community-owned.
- `creation_date`: Timestamp when the tag wiki entry was created.
- `favorite_count`: Number of times the post has been favorited.
- `last_activity_date`: Timestamp of the last activity on the post.
- `last_edit_date`: Timestamp of the last edit.
- `last_editor_display_name`: Display name of the last editor.
- `last_editor_user_id`: User ID of the last editor.
- `owner_display_name`: Display name of the owner.
- `owner_user_id`: User ID of the owner.
- `parent_id`: ID of the parent post (if applicable).
- `post_type_id`: Type of the post (e.g., 5 for Wiki entry).
- `score`: The score of the post.
- `tags`: Tags associated with the entry (often NULL in this table as it *is* a tag wiki).
- `view_count`: Number of views.

# Common query patterns

```sql
SELECT
    id,
    creation_date,
    body
  FROM
    `bigquery-public-data.stackoverflow.posts_tag_wiki`
  WHERE
    id = 5046395;
```

```sql
SELECT
    id,
    creation_date,
    last_edit_date,
    LENGTH(body) AS body_length
  FROM
    `bigquery-public-data.stackoverflow.posts_tag_wiki`
  WHERE
    creation_date > '2020-01-01 00:00:00 UTC'
  ORDER BY
    creation_date DESC
  LIMIT 10;
```

```sql
SELECT
    id,
    SUBSTR(body, 1, 100) AS body_preview
  FROM
    `bigquery-public-data.stackoverflow.posts_tag_wiki`
  WHERE
    LOWER(body) LIKE '%example code%';
```
