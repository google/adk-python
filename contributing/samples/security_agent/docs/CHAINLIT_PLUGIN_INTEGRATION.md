# Chainlit Plugin Integration Guide

This guide shows how to integrate the GCP Security Agent into your **existing Chainlit application** as a plug-and-play component.

## 🎯 Overview

The security agent is packaged as a modular `SecurityAgentProfile` class that can be dropped into any Chainlit app. Your customer can add it to their existing multi-agent Chainlit setup with minimal code changes.

## 📦 What You Get

- **4 Pre-configured Profiles**: Security Agent, Compliance Expert, Service Discovery, Documentation Search
- **32 Security Tools**: Accessed through natural language
- **Zero Conflicts**: Profile names prefixed with "GCP" to avoid naming collisions
- **Clean Integration**: No modification of existing code required

## 🚀 Quick Integration (2 Methods)

### Method 1: Simple (One-Line Integration)

```python
import chainlit as cl
from chainlit_agent import register_security_agent

# Your existing profiles
def get_my_profiles():
    return [
        cl.ChatProfile(name="My Agent 1", ...),
        cl.ChatProfile(name="My Agent 2", ...),
    ]

@cl.set_chat_profiles
async def chat_profile():
    # One line to add security agent!
    return register_security_agent(get_my_profiles())

@cl.on_chat_start
async def start():
    profile = cl.user_session.get("chat_profile")

    if SecurityAgentProfile.is_security_profile(profile):
        await SecurityAgentProfile.on_chat_start()
    else:
        # Your existing logic
        await my_existing_handler()

@cl.on_message
async def main(message: cl.Message):
    profile = cl.user_session.get("chat_profile")

    if SecurityAgentProfile.is_security_profile(profile):
        await SecurityAgentProfile.on_message(message)
    else:
        # Your existing logic
        await my_existing_handler(message)
```

### Method 2: Manual (More Control)

```python
import chainlit as cl
from chainlit_agent import SecurityAgentProfile

@cl.set_chat_profiles
async def chat_profile():
    profiles = []

    # Add your existing profiles
    profiles.append(cl.ChatProfile(name="My Agent 1", ...))
    profiles.append(cl.ChatProfile(name="My Agent 2", ...))

    # Add security agent profiles
    profiles.extend(SecurityAgentProfile.get_profiles())

    # Add more profiles...

    return profiles
```

## 📋 Step-by-Step Integration

### 1. Copy the Module

```bash
# Copy chainlit_agent.py to your Chainlit project
cp chainlit_agent.py /path/to/your/project/
```

### 2. Install Dependencies

The security agent needs these additional packages:

```bash
pip install requests python-dotenv
```

Your project already has `chainlit`, so that's all you need!

### 3. Configure Environment

Create or update `.env` file:

```bash
# Required for security agent
ADK_BASE_URL=http://localhost:8000
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/service-account.json

# Your existing environment variables...
```

### 4. Start ADK Backend

The security agent requires the ADK backend to be running:

```bash
# Terminal 1: Start ADK backend
cd /path/to/security_agent
adk web
# Runs on http://localhost:8000
```

### 5. Update Your Chainlit App

Add the routing logic to your existing app:

```python
from chainlit_agent import SecurityAgentProfile

@cl.on_chat_start
async def start():
    profile = cl.user_session.get("chat_profile")

    # NEW: Route security profiles
    if SecurityAgentProfile.is_security_profile(profile):
        await SecurityAgentProfile.on_chat_start()
        return  # Important: return after handling

    # Your existing profile handling
    if profile == "My Agent 1":
        await handle_my_agent_1()
    elif profile == "My Agent 2":
        await handle_my_agent_2()
    # ... rest of your logic

@cl.on_message
async def main(message: cl.Message):
    profile = cl.user_session.get("chat_profile")

    # NEW: Route security profiles
    if SecurityAgentProfile.is_security_profile(profile):
        await SecurityAgentProfile.on_message(message)
        return  # Important: return after handling

    # Your existing message handling
    if profile == "My Agent 1":
        await handle_my_agent_1_message(message)
    # ... rest of your logic
```

### 6. Run Your App

```bash
# Make sure ADK backend is running first!
# Then start your Chainlit app
chainlit run your_app.py
```

## 🎨 Customization

### Change Profile Names

To avoid conflicts with your existing profiles:

```python
# In chainlit_agent.py, update PROFILE_NAMES
PROFILE_NAMES = [
    "SecOps Agent",           # Instead of "GCP Security Agent"
    "SecOps Compliance",      # Instead of "GCP Compliance Expert"
    "SecOps Discovery",       # Instead of "GCP Service Discovery"
    "SecOps Documentation"    # Instead of "GCP Documentation Search"
]
```

Then update the `get_profiles()` method to match.

### Change Icons

Update icons in the `get_profiles()` method:

```python
icon="https://api.iconify.design/mdi/YOUR-ICON.svg?color=%23YOUR-COLOR"
```

Browse icons at [iconify.design](https://iconify.design)

### Customize Welcome Messages

Edit the `get_welcome_message()` method in `SecurityAgentProfile` class:

```python
agent_welcomes = {
    "GCP Security Agent": """# Your Custom Welcome Message

Your custom content here...
""",
    # ... other profiles
}
```

### Change ADK Backend URL

If your ADK backend runs on a different URL:

```python
# In .env file
ADK_BASE_URL=https://your-adk-backend.com

# Or directly in chainlit_agent.py
ADK_BASE_URL = "https://your-adk-backend.com"
```

## 🔧 Advanced: Multi-Backend Support

To support multiple backend agents:

```python
class SecurityAgentProfile:
    # Add backend selection
    BACKEND_URLS = {
        "security": "http://localhost:8000",
        "compliance": "http://localhost:8001",
        "discovery": "http://localhost:8002",
    }

    @classmethod
    def get_backend_url(cls, profile_name: str) -> str:
        """Get backend URL based on profile."""
        if "Compliance" in profile_name:
            return cls.BACKEND_URLS["compliance"]
        elif "Discovery" in profile_name:
            return cls.BACKEND_URLS["discovery"]
        else:
            return cls.BACKEND_URLS["security"]
```

## 📝 Complete Example

See [examples/chainlit_integration_example.py](../examples/chainlit_integration_example.py) for a complete working example.

## 🐛 Troubleshooting

### Security Profiles Don't Appear

**Problem**: Security profiles missing from dropdown

**Solution**: Make sure you're returning the profiles in `@cl.set_chat_profiles`:

```python
@cl.set_chat_profiles
async def chat_profile():
    profiles = your_profiles()
    profiles.extend(SecurityAgentProfile.get_profiles())
    return profiles  # Don't forget to return!
```

### Agent Not Responding

**Problem**: Agent doesn't respond to messages

**Solution**: Check that ADK backend is running:

```bash
# Check if ADK is running on port 8000
lsof -i :8000

# If not running, start it
cd /path/to/security_agent
adk web
```

### Session ID Not Found

**Problem**: Error about missing session ID

**Solution**: Make sure `on_chat_start()` is called:

```python
@cl.on_chat_start
async def start():
    profile = cl.user_session.get("chat_profile")

    if SecurityAgentProfile.is_security_profile(profile):
        await SecurityAgentProfile.on_chat_start()  # This creates the session!
```

### Profile Name Conflicts

**Problem**: Security profiles conflict with existing profiles

**Solution**: Customize profile names in `chainlit_agent.py`:

```python
PROFILE_NAMES = [
    "MyCompany Security Agent",  # Add your prefix
    # ... etc
]
```

## 🎯 Testing the Integration

### 1. Test Profile Loading

```python
# In your Chainlit app
@cl.set_chat_profiles
async def chat_profile():
    profiles = SecurityAgentProfile.get_profiles()
    print(f"Loaded {len(profiles)} security profiles")  # Should print 4
    return profiles
```

### 2. Test Profile Detection

```python
# Test the detection logic
profile = "GCP Security Agent"
is_security = SecurityAgentProfile.is_security_profile(profile)
print(f"Is security profile: {is_security}")  # Should print True
```

### 3. Test Backend Connection

```bash
# Test ADK backend is accessible
curl http://localhost:8000/apps/agents/users/web-user/sessions -X POST

# Should return JSON with session ID
```

## 🚀 Deployment

When deploying your Chainlit app with the security agent:

1. **Deploy ADK Backend First**
   ```bash
   # Deploy to Cloud Run, GKE, or your platform
   # Make sure it's accessible from your Chainlit app
   ```

2. **Update Environment Variables**
   ```bash
   # Point to production ADK backend
   ADK_BASE_URL=https://your-adk-backend.com
   ```

3. **Deploy Chainlit App**
   ```bash
   # Your normal deployment process
   # Security agent is already integrated!
   ```

## 📚 Resources

- [Chainlit Documentation](https://docs.chainlit.io/)
- [ADK Documentation](https://cloud.google.com/adk)
- [Security Agent Tools Reference](TOOLS.md)
- [Example Integration](../examples/chainlit_integration_example.py)

## ✅ Checklist for Your Customer

- [ ] Copy `chainlit_agent.py` to their project
- [ ] Install dependencies: `pip install requests python-dotenv`
- [ ] Start ADK backend: `adk web`
- [ ] Add routing logic to their `@cl.on_chat_start`
- [ ] Add routing logic to their `@cl.on_message`
- [ ] Test with `chainlit run their_app.py`
- [ ] Customize profile names if needed
- [ ] Deploy!

Your customer can now add the GCP Security Agent to their existing Chainlit app in **under 10 minutes**! 🎉
