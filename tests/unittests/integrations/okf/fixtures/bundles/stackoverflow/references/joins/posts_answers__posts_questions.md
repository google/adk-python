---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Posts Answers ↔ Posts Questions Join
description: Join path between posts_answers and posts_questions tables.
tags:
- join
- posts
- answers
- questions
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:03:04+00:00'
sources:
- id: meta_schema_doc
  title: Database schema documentation for the public data dump and SEDE
  resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
---

# posts_answers ↔ posts_questions

Join relationship between Stack Overflow questions and their answers.

```sql
ON posts_answers.parent_id = posts_questions.id
```

## Usage

Use this join path to correlate answers directly back to their parent questions to aggregate answer counts, verify metrics like Accepted Answer rate, or compare question/answer scores.

[^1]: Verified from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) on Meta Stack Exchange.
