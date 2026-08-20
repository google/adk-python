# Slack Integration

The ADK Slack integration provides a `SlackRunner` to easily deploy your agents
on Slack using [Socket Mode](https://api.slack.com/apis/connections/socket).

## Prerequisites

Install the ADK with Slack support:

```bash
pip install "google-adk[slack]"
```

## Slack App Configuration

To use the `SlackRunner`, you need to set up a Slack App in the
[Slack API Dashboard](https://api.slack.com/apps).

### 1. Enable Socket Mode
In your app settings, go to **Socket Mode** and toggle **Enable Socket Mode** to
`on`.
You will be prompted to generate an **App-Level Token** (starts with `xapp-`).
Ensure it has the `connections:write` scope.

### 2. Configure Scopes
Navigate to **OAuth & Permissions** and add the following **Bot Token Scopes**:

- `app_mentions:read`: To receive mention events.
- `chat:write`: To send messages.
- `im:history`: To respond in Direct Messages.
- `groups:history` (Optional): To respond in private channels.
- `channels:history` (Optional): To respond in public channels.

### 3. Subscribe to Events
Go to **Event Subscriptions**:

- Toggle **Enable Events** to `on`.
- Under **Subscribe to bot events**, add:
    - `app_mention`: To respond when the bot is mentioned.
    - `message.im`: To respond in Direct Messages.

### 4. Install App to Workspace
Install the app to your workspace to obtain the
**Bot User OAuth Token** (starts with `xoxb-`).

## Usage

```python
import asyncio
import os
from google.adk.runners import Runner
from google.adk.integrations.slack import SlackRunner
from slack_bolt.app.async_app import AsyncApp

async def main():
    # 1. Initialize your ADK Runner (with your agent)
    # runner = Runner(agent=my_agent, session_service=my_session_service)

    # 2. Initialize Slack AsyncApp with your Bot Token
    slack_app = AsyncApp(token=os.environ["SLACK_BOT_TOKEN"])

    # 3. Initialize the SlackRunner
    slack_runner = SlackRunner(runner=runner, slack_app=slack_app)

    # 4. Start the runner in Socket Mode with your App Token
    await slack_runner.start(app_token=os.environ["SLACK_APP_TOKEN"])

if __name__ == "__main__":
    asyncio.run(main())
```

## Session Management

The `SlackRunner` automatically manages conversation sessions:

- **Direct Messages**: The `channel_id` is used as the session ID.
- **Threaded Conversations**: The combination of `channel_id` and `thread_ts`
(the timestamp of the parent message) is used as the session ID to maintain
thread context.
- **App Mentions**: If not in a thread, the message timestamp (`ts`) is used
with the `channel_id` to start a new threaded session if the user replies
in-thread.
