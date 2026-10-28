---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Bad Question Flag Ratio
description: Calculates the ratio of spam and offensive flags (VoteTypeId 4 and 12)
  to overall votes/flags.
tags:
- metric
- votes
- moderation
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:02:44+00:00'
sources:
- id: meta_schema_doc
  resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  title: Database schema documentation for the public data dump and SEDE
---

# bad_question_flag_ratio

The bad question flag ratio measures the proportion of total flags cast on questions that are categorized as spam or offensive flags. This metric is a useful signal for tracking spam attack waves or highly inappropriate content trends.

## Formula

```sql
SAFE_DIVIDE(
  COUNTIF(VoteTypeId IN (4, 12)),
  COUNT(Id)
)
```

[^1]: Formulas sourced and derived from the `VoteTypeId` categories (4 = Offensive, 12 = Spam) documented in [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede).
