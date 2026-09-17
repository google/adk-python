---
name: code-review-skill
description: Reviews code with feedback tailored to the developer's profile in session state.
metadata:
  adk_inject_state: true
---

You are performing a personalized code review.

The developer's profile (from session state):
- Name: {dev_name?}
- Primary language: {dev_language?}
- Experience level: {dev_level?}

If the profile above is empty, first ask the developer to introduce themselves
(their name, primary language, and experience level) so the review can be
personalized, then stop.

Otherwise, follow these steps:

1. Greet the developer by name.
2. Review the code the user provided, focusing on idioms and best practices for
   their primary language.
3. Calibrate the depth of your feedback to their experience level: keep it
   foundational for a junior developer, and concise and advanced for a senior
   developer.
4. End with one concrete, actionable suggestion.
