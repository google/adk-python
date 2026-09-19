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

const CONSTANT_VALUES = require('./constant');

/**
 * Invoked from csat.yml workflow file to post survey link
 * in closed issues.
 * @param {!Object.<string,!Object>} github contains pre defined functions.
 *  context Information about the workflow run.
 * @return {null}
 */
module.exports = async ({ github, context }) => {
  const issue = context.payload.issue.html_url;

  // Check if any label matches (case-insensitive) the supported CSAT labels.
  const supportedLabels = Object.values(CONSTANT_VALUES.GLOBALS.LABELS);
  const hasMatchingLabel = context.payload.issue.labels.some(label => {
    const name = label.name.toLowerCase();
    return supportedLabels.some(supportedLabel => name.includes(supportedLabel));
  });

  if (hasMatchingLabel) {
    console.log(`Posting CSAT survey for issue =${issue}`);
    const baseUrl = CONSTANT_VALUES.MODULE.CSAT.BASE_URL;

    const yesCsat = `<a href="${baseUrl + CONSTANT_VALUES.MODULE.CSAT.SATISFACTION_PARAM +
      CONSTANT_VALUES.MODULE.CSAT.YES +
      CONSTANT_VALUES.MODULE.CSAT.ISSUEID_PARAM + encodeURIComponent(issue)}"> ${CONSTANT_VALUES.MODULE.CSAT.YES}</a>`;

    const noCsat = `<a href="${baseUrl + CONSTANT_VALUES.MODULE.CSAT.SATISFACTION_PARAM +
      CONSTANT_VALUES.MODULE.CSAT.NO +
      CONSTANT_VALUES.MODULE.CSAT.ISSUEID_PARAM + encodeURIComponent(issue)}"> ${CONSTANT_VALUES.MODULE.CSAT.NO}</a>`;

    const comment = CONSTANT_VALUES.MODULE.CSAT.MSG + '\n' + yesCsat + '\n' +
      noCsat + '\n';
    const issueNumber = context.issue.number ?? context.payload.issue.number;

    await github.rest.issues.createComment({
      issue_number: issueNumber,
      owner: context.repo.owner,
      repo: context.repo.repo,
      body: comment
    });
  }
};
