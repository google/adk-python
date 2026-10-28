---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Comments ↔ Posts Join
description: Join path between the comments and posts tables.
tags:
- join
- comments
- posts
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:02:52+00:00'
sources:
- resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  id: meta_schema_doc
  title: Database schema documentation for the public data dump and SEDE
---

# comments ↔ posts

Join relationship between the comments and posts (or answers/questions) tables.

```sql
ON comments.post_id = posts.id
```

## Usage

Use this join path to associate comment content and comment scores directly with the parent post, answer, or question. Useful for calculating comment engagement per post or finding comment threads.

[^1]: Verified from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) on Meta Stack Exchange.
