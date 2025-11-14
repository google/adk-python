import os

from dotenv import load_dotenv

# Load environment variables from a .env file for local testing
load_dotenv(override=True)

# --- GitHub API Configuration ---
GITHUB_BASE_URL = "https://api.github.com"
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
if not GITHUB_TOKEN:
  raise ValueError("GITHUB_TOKEN environment variable not set")

OWNER = os.getenv("OWNER", "google")
REPO = os.getenv("REPO", "adk-python")

STALE_LABEL_NAME = "stale"
REQUEST_CLARIFICATION_LABEL = "request clarification"

# --- THRESHOLDS IN HOURS ---
# These values can be overridden in a .env file for rapid testing (e.g., STALE_HOURS_THRESHOLD=1)
# Default: 168 hours (7 days)
# The number of hours of inactivity after a maintainer comment before an issue is marked as stale.
STALE_HOURS_THRESHOLD = float(os.getenv("STALE_HOURS_THRESHOLD", 168))

# Default: 168 hours (7 days)
# The number of hours of inactivity after an issue is marked 'stale' before it is closed.
CLOSE_HOURS_AFTER_STALE_THRESHOLD = float(
    os.getenv("CLOSE_HOURS_AFTER_STALE_THRESHOLD", 168)
)

# --- BATCH SIZE CONFIGURATION ---
# The maximum number of oldest open issues to process in a single run of the bot.
ISSUES_PER_RUN = int(os.getenv("ISSUES_PER_RUN", 100))
