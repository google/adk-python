---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_orphaned_tag_wiki
title: Orphaned Tag Wiki Posts
description: Posts that serve as wiki entries for tags that no longer exist or are
  orphaned.
tags:
- stackoverflow
- posts
- wiki
- tags
- orphaned
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:48:39+00:00'
sources:
- id: posts-orphaned-tag-wiki-resource
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_orphaned_tag_wiki
---

This table contains posts that are considered "tag wiki" entries for tags that have become orphaned or no longer exist on Stack Overflow. These posts typically provide an explanation or definition for a specific tag. The table `posts_orphaned_tag_wiki` is part of the larger [stackoverflow dataset](../datasets/stackoverflow.md).

# Schema
The schema contains fields related to the post itself, such as content, dates, and ownership information.
- `id`: Unique identifier for the post.
- `title`: The title of the post.
- `body`: The main content of the post, often in Markdown format.
- `accepted_answer_id`: (STRING) ID of the accepted answer (if applicable).
- `answer_count`: (STRING) Number of answers for the post.
- `comment_count`: Number of comments on the post.
- `community_owned_date`: Timestamp when the post became community-owned.
- `creation_date`: Timestamp when the post was created.
- `favorite_count`: (STRING) Number of times the post has been favorited.
- `last_activity_date`: Timestamp of the last activity on the post.
- `last_edit_date`: Timestamp of the last edit to the post.
- `last_editor_display_name`: Display name of the last editor.
- `last_editor_user_id`: User ID of the last editor.
- `owner_display_name`: Display name of the post owner.
- `owner_user_id`: User ID of the post owner.
- `parent_id`: (STRING) ID of the parent post (for answers or comments).
- `post_type_id`: Type of post (e.g., 1 for question, 2 for answer, 3 for tag wiki entry).
- `score`: The score of the post.
- `tags`: (STRING) Tags associated with the post (usually NULL for tag wikis themselves, as they describe the tag).
- `view_count`: (STRING) Number of views for the post.

# Common query patterns

```sql
-- Select all orphaned tag wiki posts
SELECT
    id,
    title,
    creation_date
FROM
    `bigquery-public-data.stackoverflow.posts_orphaned_tag_wiki`
LIMIT 100;
```

```sql
-- Find the body content of a specific orphaned tag wiki post
SELECT
    body
FROM
    `bigquery-public-data.stackoverflow.posts_orphaned_tag_wiki`
WHERE
    id = 4164933;
```
