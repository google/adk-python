---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_tag_wiki_excerpt
title: Posts Tag Wiki Excerpt
description: This table contains excerpt posts from the Stack Overflow tag wikis.
tags:
- stackoverflow
- tag wiki
- posts
- excerpt
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:49:51+00:00'
sources:
- id: posts-tag-wiki-excerpt-resource
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_tag_wiki_excerpt
  title: 'BigQuery Table: posts_tag_wiki_excerpt'
---

This table contains excerpt posts from the Stack Overflow tag wikis. Each row represents a summary or excerpt of a tag wiki, providing a brief description of a particular tag. This can be useful for understanding the purpose and context of various tags used on the Stack Overflow platform without needing to read the full tag wiki.

# Schema

- `id`: INTEGER, Unique identifier for the tag wiki excerpt post.
- `title`: STRING, Title of the tag wiki excerpt.
- `body`: STRING, The main content or body of the tag wiki excerpt.
- `accepted_answer_id`: STRING, ID of the accepted answer (if applicable, though unlikely for tag wiki excerpts).
- `answer_count`: STRING, Number of answers (if applicable).
- `comment_count`: INTEGER, Number of comments on the post.
- `community_owned_date`: TIMESTAMP, Date when the post became community-owned.
- `creation_date`: TIMESTAMP, Date when the post was created.
- `favorite_count`: STRING, Number of times the post has been favorited.
- `last_activity_date`: TIMESTAMP, Date of the last activity on the post.
- `last_edit_date`: TIMESTAMP, Date of the last edit to the post.
- `last_editor_display_name`: STRING, Display name of the last editor.
- `last_editor_user_id`: INTEGER, User ID of the last editor.
- `owner_display_name`: STRING, Display name of the post owner.
- `owner_user_id`: INTEGER, User ID of the post owner.
- `parent_id`: STRING, ID of the parent post (if applicable).
- `post_type_id`: INTEGER, Type of the post (e.g., 5 for Tag Wiki Excerpt).
- `score`: INTEGER, Score of the post.
- `tags`: STRING, Tags associated with the post (e.g., `<python><sql>`).
- `view_count`: STRING, Number of times the post has been viewed.

# Common query patterns

```sql
SELECT
    id,
    title,
    body
  FROM
    `bigquery-public-data.stackoverflow.posts_tag_wiki_excerpt`
  WHERE
    creation_date BETWEEN '2020-01-01' AND '2020-12-31'
  LIMIT 100;
```
```sql
SELECT
    t.tag_name,
    p.title AS excerpt_title,
    p.body AS excerpt_body
  FROM
    `bigquery-public-data.stackoverflow.posts_tag_wiki_excerpt` AS p
    JOIN `bigquery-public-data.stackoverflow.tags` AS t ON CONCAT('<', t.tag_name, '>') = p.tags
  WHERE
    t.tag_name = 'python'
  LIMIT 1;
```
