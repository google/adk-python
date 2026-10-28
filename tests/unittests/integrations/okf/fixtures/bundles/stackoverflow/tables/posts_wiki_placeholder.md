---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_wiki_placeholder
title: Stack Overflow Posts Wiki Placeholder
description: Placeholder table for various Wiki-style posts within the Stack Overflow
  dataset.
tags: stackoverflow, posts, wiki, placeholder, community
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:59:18+00:00'
sources:
- resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_wiki_placeholder
  title: 'BigQuery Public Data: Stack Overflow posts_wiki_placeholder table'
  id: stackoverflow-table
---

This table, `posts_wiki_placeholder`, serves as a repository for various Wiki-style posts found within the broader [Stack Overflow dataset](../datasets/stackoverflow.md). These posts often contain community-contributed information, guidelines, or meta-discussions rather than direct questions and answers. The content is typically informational, covering topics such as site elections, help center articles, and definitions of what constitutes a valid programming problem. Each row represents a single Wiki post, identified by a unique `id`.

# Schema
The table contains the following fields:

- `id`: Unique identifier for the post.
- `title`: Title of the post.
- `body`: The main content of the Wiki post, often containing rich text and markdown.
- `accepted_answer_id`: (NULLABLE)
- `answer_count`: (NULLABLE)
- `comment_count`: Number of comments on the post.
- `community_owned_date`: (NULLABLE)
- `creation_date`: Timestamp when the post was created.
- `favorite_count`: (NULLABLE)
- `last_activity_date`: Timestamp of the last activity on the post (e.g., edit, comment).
- `last_edit_date`: Timestamp of the last edit to the post.
- `last_editor_display_name`: Display name of the last user who edited the post.
- `last_editor_user_id`: User ID of the last user who edited the post.
- `owner_display_name`: Display name of the post's owner.
- `owner_user_id`: User ID of the post's owner. Often -1 for community-owned posts.
- `parent_id`: (NULLABLE)
- `post_type_id`: Type of the post. For this table, it consistently appears to be `7`, indicating Wiki posts.
- `score`: Score of the post, reflecting community upvotes/downvotes.
- `tags`: (NULLABLE) Tags associated with the post.
- `view_count`: (NULLABLE) Number of times the post has been viewed.

# Common query patterns

To retrieve the body content of a specific Wiki post:
```sql
SELECT
    id,
    title,
    body
  FROM
    `bigquery-public-data.stackoverflow.posts_wiki_placeholder`
  WHERE id = 8041931
```

To find the most recent Wiki posts by creation date:
```sql
SELECT
    id,
    title,
    creation_date
  FROM
    `bigquery-public-data.stackoverflow.posts_wiki_placeholder`
  ORDER BY
    creation_date DESC
  LIMIT 5
```

To count the total number of Wiki posts:
```sql
SELECT
    COUNT(DISTINCT id) AS total_wiki_posts
  FROM
    `bigquery-public-data.stackoverflow.posts_wiki_placeholder`
```
