---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/votes
title: Stack Overflow Votes
description: Records all votes cast on Stack Overflow posts.
tags: Stack Overflow, votes, posts, community
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:51:18+00:00'
sources:
- resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/votes
  title: 'BigQuery Table: votes'
  id: bq-table
- title: Database schema documentation for the public data dump and SEDE
  resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  id: meta_schema_doc
---

This table contains all individual votes cast on posts within the Stack Overflow community. Each row represents a single vote, linking it to the specific post and capturing the type of vote and when it occurred. The `vote_type_id` column can be joined with a dimension table to decode the meaning of the vote (e.g., upvote, downvote, favorite).

For a detailed catalog of the `vote_type_id` column values, see the [Vote Types Reference](../references/vote_types.md). [^1]

# Schema

*   `id`: Unique identifier for the vote.
*   `creation_date`: The date and time when the vote was cast. (Note: Time information is truncated to `00:00:00` for privacy). [^1]
*   `post_id`: The identifier of the post that received the vote. This can be joined with the `id` column in tables such as [posts_questions](posts_questions.md) or [posts_answers](posts_answers.md) to get details about the voted post.
*   `vote_type_id`: The type of vote cast (e.g., upvote, downvote, favorite). See [Vote Types Reference](../references/vote_types.md). [^1]

# Common query patterns

To count the total number of upvotes for a specific post:
```sql
SELECT
    count(*) AS upvotes
  FROM
    `bigquery-public-data.stackoverflow.votes`
  WHERE
    post_id = 12345 -- Replace with an actual post ID
    AND vote_type_id = 2 -- Assuming '2' represents an upvote
```

To find the top 10 most voted posts by total votes:
```sql
SELECT
    post_id,
    count(*) AS total_votes
  FROM
    `bigquery-public-data.stackoverflow.votes`
  GROUP BY
    post_id
  ORDER BY
    total_votes DESC
  LIMIT 10
```

To see the distribution of vote types:
```sql
SELECT
    vote_type_id,
    count(*) AS vote_count
  FROM
    `bigquery-public-data.stackoverflow.votes`
  GROUP BY
    vote_type_id
  ORDER BY
    vote_count DESC
```

# Metrics

- [Bad Question Flag Ratio](../references/metrics/bad_question_flag_ratio.md) — Calculates the ratio of spam and offensive flags on questions. [^1]

# Joins

- [posts](../references/joins/posts__votes.md) — join on `post_id` ↔ `id` to associate vote actions with questions, answers, and other post types. [^1]

[^1]: Sourced from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) on Meta Stack Exchange.
