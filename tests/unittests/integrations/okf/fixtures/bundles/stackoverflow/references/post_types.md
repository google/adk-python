---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Post Types Reference
description: Enum lookup values for the PostTypeId column in Stack Overflow posts
  tables.
tags:
- lookup
- enum
- posts
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:02:30+00:00'
sources:
- resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  title: Database schema documentation for the public data dump and SEDE
  id: meta_schema_doc
- id: meta_post_types
  title: Meaning of Values for PostTypeId in data explorer or in data-dump
  resource: https://meta.stackexchange.com/questions/99265/meaning-of-values-for-posttypeid-in-data-explorer-or-in-data-dump
---

# Post Types Reference

A lookup catalog defining the meaning of the `PostTypeId` attribute used in the primary `posts_answers`, `posts_questions`, `posts_moderator_nomination`, `posts_orphaned_tag_wiki`, `posts_privilege_wiki`, `posts_tag_wiki`, `posts_tag_wiki_excerpt`, `posts_wiki_placeholder`, and `stackoverflow_posts` tables.

## Lookup Catalog

| PostTypeId | Name | Description |
|---|---|---|
| 1 | Question | A user-submitted question. |
| 2 | Answer | A user-submitted answer to a question. |
| 3 | Orphaned tag wiki | Tag wikis for tags that have since been deleted. |
| 4 | Tag wiki excerpt | Short intro/excerpt text for a tag. |
| 5 | Tag wiki | Full body text detailing a tag's usage guidelines. |
| 6 | Moderator nomination | Moderator candidate nomination posts. |
| 7 | Wiki placeholder | Auxiliary site content (e.g. Help Center intro, election description, tour intro). |
| 8 | Privilege wiki | Privilege description pages. |
| 9 | Article | Article post type. |
| 10 | HelpArticle | Help Center articles. |
| 12 | Collection | Content collections. |
| 13 | ModeratorQuestionnaireResponse | Candidate answers to moderator questionnaires. |
| 14 | Announcement | Site announcements. |
| 15 | CollectiveDiscussion | Stack Overflow Collectives discussion threads. |
| 17 | CollectiveCollection | Stack Overflow Collectives collections. |

[^1]: Verified from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) and [PostTypeId Meanings](https://meta.stackexchange.com/questions/99265/meaning-of-values-for-posttypeid-in-data-explorer-or-in-data-dump) on Meta Stack Exchange.
