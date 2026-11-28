---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_questions
title: Stack Overflow Posts Questions
description: Contains all question posts from Stack Overflow.
tags: stackoverflow, posts, questions
generated:
  at: '2026-07-10T22:49:19+00:00'
  by: reference_agent/gemini-2.5-flash
sources:
- id: stackoverflow-posts-questions
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_questions
  title: Stack Overflow Posts Questions Table
- id: meta_schema_doc
  resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  title: Database schema documentation for the public data dump and SEDE
---

This table contains all question posts from Stack Overflow, including their content, metadata, and various counts related to activity and user engagement. Each row represents a unique question post. This table can be joined with other tables like [posts_answers](posts_answers.md) to link questions to their corresponding answers, or with [users](users.md) to get more information about the question's owner.

For a detailed catalog of the `post_type_id` column values, see the [Post Types Reference](../references/post_types.md). For details on post licensing, see the [Creative Commons Content Licenses Reference](../references/content_licenses.md). [^1]

# Schema

- `id`: INTEGER, Unique identifier for the post.
- `title`: STRING, The title of the question.
- `body`: STRING, The main content of the question.
- `accepted_answer_id`: INTEGER, The ID of the accepted answer, if any.
- `answer_count`: INTEGER, Number of answers for the question.
- `comment_count`: INTEGER, Number of comments on the question.
- `community_owned_date`: TIMESTAMP, Date when the question became community owned.
- `creation_date`: TIMESTAMP, Date and time when the question was created.
- `favorite_count`: INTEGER, Number of times the question has been favorited.
- `last_activity_date`: TIMESTAMP, Date and time of the last activity on the question.
- `last_edit_date`: TIMESTAMP, Date and time of the last edit to the question.
- `last_editor_display_name`: STRING, Display name of the last editor.
- `last_editor_user_id`: INTEGER, User ID of the last editor.
- `owner_display_name`: STRING, Display name of the question's owner.
- `owner_user_id`: INTEGER, User ID of the question's owner (links to [users](users.md)).
- `parent_id`: STRING, Parent ID (not typically used for questions).
- `post_type_id`: INTEGER, Type of post (1 for question). See [Post Types Reference](../references/post_types.md). [^1]
- `score`: INTEGER, The score of the question.
- `tags`: STRING, Tags associated with the question, separated by '|'.
- `view_count`: INTEGER, Number of times the question has been viewed.

# Common query patterns

```sql
SELECT
    id,
    title,
    view_count
FROM
    `bigquery-public-data.stackoverflow.posts_questions`
ORDER BY
    view_count DESC
LIMIT 10;
```

```sql
SELECT
    p.title,
    p.score,
    u.display_name AS owner_name
FROM
    `bigquery-public-data.stackoverflow.posts_questions` AS p
JOIN
    `bigquery-public-data.stackoverflow.users` AS u
ON
    p.owner_user_id = u.id
WHERE
    p.creation_date BETWEEN '2022-01-01' AND '2022-01-31'
ORDER BY
    p.score DESC
LIMIT 5;
```

```sql
SELECT
    EXTRACT(DATE FROM creation_date) AS question_date,
    COUNT(id) AS number_of_questions
FROM
    `bigquery-public-data.stackoverflow.posts_questions`
GROUP BY
    question_date
ORDER BY
    question_date DESC;
```

# Metrics

- [Accepted Answer Rate](../references/metrics/accepted_answer_rate.md) — Calculates the proportion of questions having an accepted answer. [^1]

# Joins

- [posts_answers](../references/joins/posts_answers__posts_questions.md) — join on `id` ↔ `parent_id` to attach answers to questions. [^1]
- [comments](../references/joins/comments__posts.md) — join on `id` ↔ `post_id` to correlate comments with questions. [^1]
- [votes](../references/joins/posts__votes.md) — join on `id` ↔ `post_id` to find votes/flags on questions. [^1]
- [post_links](../references/joins/post_links__posts.md) — join on `id` ↔ `post_id` to discover duplicate and related question links. [^1]

[^1]: Sourced from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) on Meta Stack Exchange.
