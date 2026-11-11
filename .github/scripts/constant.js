/*
Copyright 2026 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

let CONSTANT_VALUES = {
  GLOBALS: {
    LABELS: {
      BUG: 'bug',
      CORE: 'core',
      TOOLS: 'tools',
      SERVICES: 'services',
      MODELS: 'models',
      MCP: 'mcp',
      AUTH: 'auth',
      LIVE: 'live',
      DOCUMENTATION: 'documentation',
      GOOD_FIRST_ISSUE: 'good first issue',
      AGENT_ENGINE: 'agent engine',
      BQ: 'bq',
      EVAL: 'eval',
      TRACING: 'tracing',
      WEB: 'web',
      WORKFLOW: 'workflow'
    },
    STATE: { CLOSED: 'closed' }
  },
  MODULE: {
    CSAT: {
      YES: 'Yes',
      NO: 'No',
      BASE_URL:
        'https://docs.google.com/forms/d/e/1FAIpQLScgyeKPxUlq4kgNuI7g9_iXkQKlzT6ZvGA656x5HpbUpYjOsg/viewform?usp=pp_url&',
      SATISFACTION_PARAM: 'entry.817493361=',
      ISSUEID_PARAM: '&entry.1977942008=',
      MSG: 'Are you satisfied with the resolution of your issue?',
    }
  }

};
module.exports = CONSTANT_VALUES;
