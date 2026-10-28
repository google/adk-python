---
type: BigQuery Table
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/stackoverflow_posts
title: Stack Overflow Posts (Deprecated)
description: A deprecated table containing Stack Overflow posts. Use the posts_answers
  or posts_questions tables instead.
tags: stackoverflow, posts, deprecated
status: deprecated
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:50:28+00:00'
sources:
- title: Deprecated Stack Overflow Posts Table
  id: stackoverflow-posts-table
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow/tables/stackoverflow_posts
---

This table, `stackoverflow_posts`, contains a comprehensive collection of posts from Stack Overflow. Each row represents a single post, which can be a question, an answer, or another type of post. Key information includes the post's title, body, creation date, score, and associated tags.

**WARNING:** This table is **deprecated** and should not be used for new development or analysis. For up-to-date and more specialized data, please use the individual post type tables, specifically [posts_questions](posts_questions.md) for questions and [posts_answers](posts_answers.md) for answers.

# Schema
The `stackoverflow_posts` table contains the following fields:

*   `id`: `INTEGER` (REQUIRED) - Unique identifier for the post.
*   `title`: `STRING` - The title of the post (e.g., for questions).
*   `body`: `STRING` - The main content of the post.
*   `accepted_answer_id`: `INTEGER` - The ID of the accepted answer, if applicable.
*   `answer_count`: `INTEGER` - Number of answers to a question.
*   `comment_count`: `INTEGER` - Number of comments on the post.
*   `community_owned_date`: `TIMESTAMP` - Date when the post became community owned.
*   `creation_date`: `TIMESTAMP` - Date and time the post was created.
*   `favorite_count`: `INTEGER` - Number of times the post has been favorited.
*   `last_activity_date`: `TIMESTAMP` - Last date of activity on the post.
*   `last_edit_date`: `TIMESTAMP` - Last date the post was edited.
*   `last_editor_display_name`: `STRING` - Display name of the last editor.
*   `last_editor_user_id`: `INTEGER` - User ID of the last editor.
*   `owner_display_name`: `STRING` - Display name of the post owner.
*   `owner_user_id`: `INTEGER` - User ID of the post owner.
*   `parent_id`: `INTEGER` - For answers, the ID of the question it answers.
*   `post_type_id`: `INTEGER` - Type of the post (e.g., 1 for Question, 2 for Answer).
*   `score`: `INTEGER` - The current score of the post.
*   `tags`: `STRING` - Tags associated with the post, typically for questions (e.g., `<python><django>`).
*   `view_count`: `INTEGER` - Number of times the post has been viewed.

# Common query patterns

```sql
-- DANGER: This table is deprecated. Do not use for new queries.
-- Example of selecting basic post information (for historical context only).
SELECT
    id,
    title,
    creation_date,
    score,
    tags
FROM
    `bigquery-public-data.stackoverflow.stackoverflow_posts`
WHERE
    creation_date BETWEEN '2016-01-01' AND '2016-01-31'
LIMIT 100;
```
