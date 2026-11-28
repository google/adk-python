---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Vote Types Reference
description: Enum lookup values for the VoteTypeId column in the Stack Overflow votes
  table.
tags:
- lookup
- enum
- votes
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:02:36+00:00'
sources:
- id: meta_schema_doc
  resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  title: Database schema documentation for the public data dump and SEDE
- title: List of Vote type IDs
  resource: https://meta.stackexchange.com/questions/171176/list-of-vote-type-ids
  id: meta_vote_types
---

# Vote Types Reference

A lookup catalog defining the meaning of the `VoteTypeId` attribute in the `votes` table.

## Lookup Catalog

| VoteTypeId | Name | Description |
|---|---|---|
| -1 | InformModerator | Flag raised to bring a moderator's attention to a post. |
| 0 | UndoMod | Undo a moderation action or vote. |
| 1 | AcceptedByOriginator | Question owner accepted an answer. |
| 2 | UpMod | Question/Answer upvote. |
| 3 | DownMod | Question/Answer downvote. |
| 4 | Offensive | Flagged as offensive or abusive. |
| 5 | Favorite | Bookmark (now deprecated and replaced by Saves). |
| 6 | Close | Vote to close a question. (No longer populated here; close votes reside in PostHistory). |
| 7 | Reopen | Vote to reopen a question. |
| 8 | BountyStart | User started a bounty on a question. |
| 9 | BountyClose | Bounty closed/awarded on a question. |
| 10 | Deletion | Vote to delete a post. |
| 11 | Undeletion | Vote to undelete a post. |
| 12 | Spam | Flagged as spam. |
| 15 | ModeratorReview | A moderator reviewed a flagged post. |
| 16 | ApproveEditSuggestion | Vote to approve a suggested edit. |
| 17-28 | Teams Reactions | Reactions (e.g. celebrate, smile, heart) implemented in Stack Overflow for Teams. |
| 29 | Outdated | Answer flagged as outdated. |
| 30 | NotOutdated | Vote asserting an answer is not outdated. |
| 31 | PreVote | Pre-vote action. |
| 32 | CollectiveDiscussionUpvote | Upvote on a Collectives discussion. |
| 33 | CollectiveDiscussionDownvote | Downvote on a Collectives discussion (deprecated). |
| 35 | privateAiAnswerCorrect | Vote stating AI answer is correct (experiment). |
| 36 | privateAiAnswerIncorrect | Vote stating AI answer is incorrect (experiment). |
| 37 | privateAiAnswerPartiallyCorrect | Vote stating AI answer is partially correct. |

[^1]: Verified from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) and [List of Vote type IDs](https://meta.stackexchange.com/questions/171176/list-of-vote-type-ids) on Meta Stack Exchange.
