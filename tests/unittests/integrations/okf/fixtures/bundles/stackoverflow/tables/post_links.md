---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/post_links
title: Post Links
description: Contains information about links between posts on Stack Overflow.
tags:
- stackoverflow
- posts
- links
- cross-references
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:47:44+00:00'
sources:
- id: post_links_table
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/post_links
---

The `post_links` table stores information about how posts on Stack Overflow are linked to each other. Each row represents a single link between two posts, indicating a relationship such as a duplicate question, a related question, or a wiki link. This table is crucial for understanding the interconnectedness of content within the Stack Overflow platform.

The grain of this table is one row per link between two posts.

# Schema

- `id`: Unique identifier for the post link.
- `creation_date`: The date and time when the link was created.
- `link_type_id`: Identifier for the type of link (e.g., duplicate, related).
- `post_id`: The ID of the primary post in the link. This typically refers to a post in the [posts_questions](posts_questions.md) or [posts_answers](posts_answers.md) tables.
- `related_post_id`: The ID of the related post in the link, also referring to a post in the [posts_questions](posts_questions.md) or [posts_answers](posts_answers.md) tables.

# Common query patterns

```sql
SELECT
  pl.id,
  pl.creation_date,
  pl.link_type_id,
  p1.title AS post_title,
  p2.title AS related_post_title
FROM
  `bigquery-public-data.stackoverflow.post_links` AS pl
JOIN
  `bigquery-public-data.stackoverflow.posts_questions` AS p1
  ON pl.post_id = p1.id
JOIN
  `bigquery-public-data.stackoverflow.posts_questions` AS p2
  ON pl.related_post_id = p2.id
WHERE
  pl.link_type_id = 3 -- Example: LinkType = "Related"
LIMIT 100;
```

```sql
SELECT
  link_type_id,
  COUNT(*)
FROM
  `bigquery-public-data.stackoverflow.post_links`
GROUP BY
  link_type_id
ORDER BY
  COUNT(*) DESC;
```
