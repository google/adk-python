---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Posts ↔ Votes Join
description: Join path between the votes and posts tables.
tags:
- join
- posts
- votes
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:02:55+00:00'
sources:
- resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  id: meta_schema_doc
  title: Database schema documentation for the public data dump and SEDE
---

# posts ↔ votes

Join relationship between the votes table and the posts tables.

```sql
ON votes.post_id = posts.id
```

## Usage

Use this join path to associate individual votes, flags, and favorites with their target posts (questions, answers, or moderator nominations).

[^1]: Verified from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) on Meta Stack Exchange.
