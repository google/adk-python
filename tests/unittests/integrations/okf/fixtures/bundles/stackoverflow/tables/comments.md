---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/comments
title: Comments
description: Contains all comments made on posts within the Stack Overflow dataset.
tags:
- comments
- stackoverflow
- user activity
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:47:09+00:00'
sources:
- id: stackoverflow-comments
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/comments
  title: Stack Overflow Comments Table
---

The `comments` table within the [Stack Overflow dataset](../datasets/stackoverflow.md) contains a record of all comments posted on questions and answers by users. Each row represents a single comment, providing details such as the comment's text, its creation date, the associated post, and the user who made the comment. This table can be joined with the [posts_questions](posts_questions.md) or [posts_answers](posts_answers.md) tables on `post_id` to retrieve the content being commented on, and with the [users](users.md) table on `user_id` to get more information about the commenter. The data spans from September 2008 onwards.

# Schema

- `id`: Unique identifier for the comment.
- `text`: The content of the comment.
- `creation_date`: Timestamp when the comment was created.
- `post_id`: The ID of the post (question or answer) the comment belongs to.
- `user_id`: The ID of the user who made the comment.
- `user_display_name`: The display name of the user who made the comment (may be NULL if user is anonymous or deleted).
- `score`: The score or upvotes received by the comment.

# Common query patterns

1.  **Retrieve all comments for a specific post:**
    ```sql
    SELECT
      id,
      text,
      creation_date,
      user_display_name,
      score
    FROM
      `bigquery-public-data.stackoverflow.comments`
    WHERE
      post_id = 47885
    ORDER BY
      creation_date DESC
    LIMIT 100;
    ```

2.  **Count comments per user:**
    ```sql
    SELECT
      user_display_name,
      COUNT(id) AS total_comments
    FROM
      `bigquery-public-data.stackoverflow.comments`
    WHERE
      user_display_name IS NOT NULL
    GROUP BY
      user_display_name
    ORDER BY
      total_comments DESC
    LIMIT 10;
    ```

3.  **Find comments on questions containing a specific keyword:**
    ```sql
    SELECT
      c.id,
      c.text AS comment_text,
      q.title AS question_title,
      q.body AS question_body,
      c.creation_date
    FROM
      `bigquery-public-data.stackoverflow.comments` AS c
    JOIN
      `bigquery-public-data.stackoverflow.posts_questions` AS q
    ON
      c.post_id = q.id
    WHERE
      q.title LIKE '%python%'
    ORDER BY
      c.creation_date DESC
    LIMIT 10;
    ```
