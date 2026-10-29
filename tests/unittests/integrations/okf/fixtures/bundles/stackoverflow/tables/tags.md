---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/tags
title: Tags
description: Contains information about tags used on Stack Overflow, including their
  names and usage counts.
tags:
- stackoverflow
- tags
- metadata
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:50:47+00:00'
sources:
- resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/tags
  title: Stack Overflow Tags BigQuery Table
  id: stackoverflow-tags-table
- id: stackoverflow-website
  resource: https://stackoverflow.com/
  title: Stack Overflow Website
---

The `tags` table in the [stackoverflow](../datasets/stackoverflow.md) dataset provides a comprehensive list of all tags used across Stack Overflow, along with their associated metadata. Each row represents a unique tag, detailing its identifier, name, the number of times it has been used, and references to its excerpt and wiki posts. This table is essential for understanding the categorization of questions and answers within the Stack Overflow community.

# Schema

- `id`: Unique identifier for the tag. (INTEGER)
- `tag_name`: The name of the tag (e.g., 'python', 'java', 'c#'). (STRING)
- `count`: The total number of times this tag has been used. (INTEGER)
- `excerpt_post_id`: The ID of the post containing the tag's excerpt description. This can be joined with [posts_tag_wiki_excerpt](posts_tag_wiki_excerpt.md) or [posts_tag_wiki](posts_tag_wiki.md) tables. (INTEGER)
- `wiki_post_id`: The ID of the post containing the tag's full wiki description. This can be joined with [posts_tag_wiki](posts_tag_wiki.md) table. (INTEGER)

# Common query patterns

```sql
-- Retrieve the top 10 most used tags
SELECT
    tag_name,
    count
  FROM
    `bigquery-public-data.stackoverflow.tags`
  ORDER BY
    count DESC
  LIMIT 10;
```

```sql
-- Find details for a specific tag
SELECT
    id,
    tag_name,
    count,
    excerpt_post_id,
    wiki_post_id
  FROM
    `bigquery-public-data.stackoverflow.tags`
  WHERE
    tag_name = 'python';
```

```sql
-- Count total unique tags
SELECT
    COUNT(DISTINCT tag_name) AS total_unique_tags
  FROM
    `bigquery-public-data.stackoverflow.tags`;
```
