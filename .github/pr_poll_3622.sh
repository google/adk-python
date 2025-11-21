#!/usr/bin/env bash
set -euo pipefail

PR_NUMBER=3622
REPO=google/adk-python
LOG=/workspaces/adk-python/.pr_poll_3622.log

echo "Starting PR poller for $REPO#$PR_NUMBER, logging to $LOG"

# Ensure log exists
mkdir -p "$(dirname "$LOG")"
: > "$LOG"

while true; do
  ts=$(date --iso-8601=seconds)
  if command -v gh >/dev/null 2>&1; then
    # Try to get a concise status line with gh; fallback to full JSON on failure
    if out=$(gh pr view "$PR_NUMBER" --repo "$REPO" --json number,title,author,headRefName,baseRefName,mergeStateStatus 2>&1); then
      echo "[$ts] $out" >> "$LOG"
      # append a short status
      echo "[$ts] SHORT: $(echo "$out" | head -n 1)" >> "$LOG"
    else
      echo "[$ts] GH_ERROR: $out" >> "$LOG"
    fi
  elif [[ -n "${GITHUB_TOKEN:-}" ]]; then
    out=$(curl -s -H "Authorization: token $GITHUB_TOKEN" "https://api.github.com/repos/$REPO/pulls/$PR_NUMBER") || out="ERROR_FROM_CURL"
    echo "[$ts] $out" >> "$LOG"
    echo "[$ts] SHORT: $(echo "$out" | head -c 200)" >> "$LOG"
  else
    echo "[$ts] ERROR: gh CLI not available and GITHUB_TOKEN not set" >> "$LOG"
  fi

  # NOTE: do not exit automatically here; keep polling until manually stopped.
  # If desired we could exit when the PR is merged/closed, but some CI checks
  # remain visible after merge, so prefer continuous monitoring.

  sleep 600
done
