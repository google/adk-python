---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Post Links ↔ Posts Join
description: Join path between the post_links and posts tables.
tags:
- join
- posts
- links
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:03:01+00:00'
sources:
- title: Database schema documentation for the public data dump and SEDE
  id: meta_schema_doc
  resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
---

# post_links ↔ posts

Join relationship between the post links table and target/source posts.

```sql
ON post_links.post_id = posts.id
```

## Usage

Use this join path to resolve metadata for the source post (`post_id`) or targets (`related_post_id`) in a link/duplicate relationship.

[^1]: Verified from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) on Meta Stack Exchange.
