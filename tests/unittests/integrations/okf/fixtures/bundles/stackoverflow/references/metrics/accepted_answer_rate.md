---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Accepted Answer Rate
description: The proportion of questions that have an accepted answer.
tags:
- metric
- posts
- community
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:02:48+00:00'
sources:
- resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  id: meta_schema_doc
  title: Database schema documentation for the public data dump and SEDE
---

# accepted_answer_rate

The accepted answer rate measures the proportion of questions that have a resolved and accepted answer. It is a core community-health KPI indicating question resolution efficiency.

## Formula

```sql
SAFE_DIVIDE(
  COUNT(AcceptedAnswerId),
  COUNT(Id)
)
```

[^1]: Formulas sourced and derived from the `posts_questions` and `stackoverflow_posts` schema documented in [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede).
