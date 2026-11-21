---
type: BigQuery Dataset
resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow
title: Stack Overflow Public Dataset
description: This dataset contains a public archive of Stack Overflow data, including
  posts, users, and tags. It was last updated on 2022-11-25 and is no longer actively
  updated.
tags: Stack Overflow, Q&A, developer, programming, public dataset
generated:
  by: reference_agent/gemini-2.5-flash
  at: '2026-07-10T22:46:36+00:00'
sources:
- title: Stack Overflow Public Dataset
  resource: https://bigquery.googleapis.com/v2/projects/bigquery-public-data/datasets/stackoverflow
  id: stackoverflow-dataset-resource
---

The `stackoverflow` dataset, hosted in BigQuery's public data program, provides a comprehensive archive of Stack Overflow's community-generated content. It includes information on questions, answers, comments, users, badges, and tags, offering a rich resource for analyzing developer activity, programming trends, and community dynamics. The data was last updated on 2022-11-25 and is no longer actively updated by its original source. It is located in the `US` multi-region.

# Schema
This dataset contains the following tables:

*   [`badges`](../tables/badges.md): Information about badges awarded to users.
*   [`comments`](../tables/comments.md): User-submitted comments on posts.
*   [`post_history`](../tables/post_history.md): Historical revisions and events for posts.
*   [`post_links`](../tables/post_links.md): Links between posts.
*   [`posts_answers`](../tables/posts_answers.md): Answers to questions.
*   [`posts_moderator_nomination`](../tables/posts_moderator_nomination.md): Posts related to moderator nominations.
*   [`posts_orphaned_tag_wiki`](../tables/posts_orphaned_tag_wiki.md): Orphaned tag wiki posts.
*   [`posts_privilege_wiki`](../tables/posts_privilege_wiki.md): Privilege wiki posts.
*   [`posts_questions`](../tables/posts_questions.md): User-submitted questions.
*   [`posts_tag_wiki`](../tables/posts_tag_wiki.md): Tag wiki entries.
*   [`posts_tag_wiki_excerpt`](../tables/posts_tag_wiki_excerpt.md): Excerpts from tag wiki entries.
*   [`posts_wiki_placeholder`](../tables/posts_wiki_placeholder.md): Placeholder posts for wiki content.
*   [`stackoverflow_posts`](../tables/stackoverflow_posts.md): A consolidated view of all posts (questions and answers).
*   [`tags`](../tables/tags.md): Information about tags used on Stack Overflow.
*   [`users`](../tables/users.md): User profiles and statistics.
*   [`votes`](../tables/votes.md): Records of votes on posts.

# Common query patterns
To explore the tables within this dataset:

```sql
SELECT table_name
FROM `bigquery-public-data.stackoverflow.INFORMATION_SCHEMA.TABLES`
WHERE table_schema = 'stackoverflow';
```

To query the number of questions posted in a specific year:

```sql
SELECT
  EXTRACT(YEAR FROM creation_date) AS year,
  COUNT(*) AS num_questions
FROM `bigquery-public-data.stackoverflow.posts_questions`
GROUP BY 1
ORDER BY 1 DESC
LIMIT 100;
```
