---
type: Reference
resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
title: Creative Commons Content Licenses Reference
description: Lookup table defining user-contributed content licensing rules and dates
  on the Stack Exchange network.
tags:
- license
- legal
- meta
generated:
  by: reference_agent/gemini-3.5-flash
  at: '2026-07-10T23:02:40+00:00'
sources:
- resource: https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede
  id: meta_schema_doc
  title: Database schema documentation for the public data dump and SEDE
---

# Creative Commons Content Licenses

Lookup table defining Stack Overflow user content licensing over time based on the `ContentLicense` attribute.

## Content License Lookups

| ContentLicense Value | Date Start | Date End | License Link |
| --- | --- | --- | --- |
| **CC BY-SA 4.0** | 2018-05-02 | *present* | [Creative Commons BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) |
| **CC BY-SA 3.0** | 2011-04-08 | 2018-05-01 | [Creative Commons BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/) |
| **CC BY-SA 2.5** | *inception* | 2011-04-07 | [Creative Commons BY-SA 2.5](https://creativecommons.org/licenses/by-sa/2.5/) |

[^1]: Verified from [Database Schema Documentation](https://meta.stackexchange.com/questions/2677/database-schema-documentation-for-the-public-data-dump-and-sede) on Meta Stack Exchange.
