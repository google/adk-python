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

# --- Agent Logic Configuration ---
MAINTAINERS_STR = os.getenv("MAINTAINERS", "")
if not MAINTAINERS_STR:
    raise ValueError("MAINTAINERS environment variable not set. Please provide a comma-separated list of GitHub usernames.")
# Parse the comma-separated string into a Python list
MAINTAINERS = [m.strip() for m in MAINTAINERS_STR.split(',') if m.strip()]

STALE_LABEL_NAME = "stale"

# --- THRESHOLDS IN HOURS ---
# These values can be overridden in a .env file for rapid testing (e.g., STALE_HOURS_THRESHOLD=1)

# Default: 168 hours (7 days)
# The number of hours of inactivity after a maintainer comment before an issue is marked as stale.
STALE_HOURS_THRESHOLD = int(os.getenv("STALE_HOURS_THRESHOLD", 168))

# Default: 168 hours (7 days)
# The number of hours of inactivity after an issue is marked 'stale' before it is closed.
CLOSE_HOURS_AFTER_STALE_THRESHOLD = int(os.getenv("CLOSE_HOURS_AFTER_STALE_THRESHOLD", 168))