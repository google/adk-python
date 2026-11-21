---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_answers
title: Posts Answers
description: Contains Stack Overflow answers, including their content, scores, and
  associated metadata.
tags: stackoverflow, answers, posts, Q&A
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:48:04+00:00'
sources:
- title: Stack Overflow Posts Answers Table
  id: stackoverflow-posts_answers
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/posts_answers
---

The `posts_answers` table in the `bigquery-public-data.stackoverflow` dataset contains all answers submitted by users on the Stack Overflow platform. Each row in this table represents a single answer to a question. Key information includes the answer's `body` (content), `creation_date`, `score`, and `owner_user_id`. Answers are linked to their corresponding questions via the `parent_id` field, which references the `id` from the [posts_questions](posts_questions.md) table. This table is useful for analyzing answer quality, user contributions, and trends in responses over time.

# Schema

- id: INTEGER (Unique ID of the answer)
- title: STRING (Title of the post. Typically NULL for answers, as the title belongs to the question)
- body: STRING (The HTML content of the answer)
- accepted_answer_id: STRING (ID of the accepted answer for the parent question. Typically NULL for answers themselves)
- answer_count: STRING (Number of answers for the parent question. Typically NULL for answers)
- comment_count: INTEGER (Number of comments on this specific answer)
- community_owned_date: TIMESTAMP (Date when the answer became community-owned)
- creation_date: TIMESTAMP (UTC timestamp when the answer was posted)
- favorite_count: STRING (Number of times the parent question was favorited. Typically NULL for answers)
- last_activity_date: TIMESTAMP (UTC timestamp of the last activity on this answer)
- last_edit_date: TIMESTAMP (UTC timestamp of the last edit to this answer)
- last_editor_display_name: STRING (Display name of the user who last edited the answer)
- last_editor_user_id: INTEGER (User ID of the user who last edited the answer)
- owner_display_name: STRING (Display name of the user who posted the answer)
- owner_user_id: INTEGER (User ID of the user who posted the answer)
- parent_id: INTEGER (The ID of the question this answer belongs to. Links to `id` in the [posts_questions](posts_questions.md) table.)
- post_type_id: INTEGER (The type of post; `2` for answers.)
- score: INTEGER (The current score of the answer, based on upvotes and downvotes)
- tags: STRING (Tags associated with the parent question. Typically NULL for answers)
- view_count: STRING (View count of the parent question. Typically NULL for answers)

# Common query patterns

```sql
SELECT
  id,
  body,
  score,
  creation_date
FROM
  `bigquery-public-data.stackoverflow.posts_answers`
ORDER BY
  score DESC
LIMIT 10
```

```sql
SELECT
  owner_user_id,
  count(id) AS answer_count
FROM
  `bigquery-public-data.stackoverflow.posts_answers`
WHERE
  owner_user_id IS NOT NULL
GROUP BY
  owner_user_id
ORDER BY
  answer_count DESC
LIMIT 5
```

```sql
SELECT
  id,
  body,
  score,
  owner_display_name
FROM
  `bigquery-public-data.stackoverflow.posts_answers`
WHERE
  parent_id = 12345
ORDER BY
  score DESC
```
