# Quick Start Guide - GCP Security Agent

## 🚀 For New Users (No GCP Credentials Needed)

This guide helps you get the security agent running quickly with demo data, perfect for testing and learning.

### Step 1: Clone and Setup

```bash
git clone <your-repo-url>
cd security_agent
```

### Step 2: Install Dependencies

```bash
# Install Python dependencies
pip install -r requirements.txt
pip install -r requirements_frontend.txt

# Or create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -r requirements_frontend.txt
```

### Step 3: Setup Demo Data (No GCP Needed)

```bash
# Create demo database with test data
python scripts/setup_demo_data.py
```

This creates a SQLite database with:
- ✅ 14 storage buckets with realistic security configurations
- ✅ 5 security findings (public buckets, unencrypted data, etc.)
- ✅ 5 compute instances
- ✅ 5 IAM accounts
- ✅ 3 networks and 4 firewall rules
- ✅ 3 databases

### Step 4: Configure Environment

```bash
# Copy the example environment file
cp .env.example .env

# Edit .env to use demo settings (already configured for demo mode)
# No changes needed - defaults work with demo data!
```

### Step 5: Start Services

**Option A: Using the startup script (recommended)**
```bash
chmod +x scripts/start_services.sh
./scripts/start_services.sh
```

**Option B: Manual startup**
```bash
# Terminal 1: Start Backend
python run_backend.py

# Terminal 2: Start Frontend (new terminal)
python run_frontend.py
```

### Step 6: Access the Application

- 🖥️ **Frontend UI**: http://localhost:8501
- 📡 **Backend API**: http://localhost:8000
- 📚 **API Documentation**: http://localhost:8000/docs

### Step 7: Test the Agent

Try these example queries in the chat interface:

1. **"What are my biggest security risks and how should I prioritize fixing them?"**
   - Should return LLM analysis of your security findings

2. **"Analyze my storage buckets and recommend security improvements"**
   - Should provide detailed analysis of the 14 demo buckets

3. **"Which security findings pose the highest risk to my organization?"**
   - Should prioritize the 5 demo security findings

4. **"How can I improve my overall GCP security stance?"**
   - Should provide comprehensive security recommendations

### Expected Behavior

The agent should respond with **LLM-generated analysis**, not raw JSON data. You should see:
- ✅ Custom insights and recommendations
- ✅ Prioritized action items
- ✅ Context-aware security advice
- ❌ **NOT** raw JSON like `{"success": true, "data": [...]}`

---

## 🏢 For Users with GCP Credentials

If you have a real GCP project and want to use live data:

### Step 1: Setup GCP Authentication

```bash
# Option A: Use gcloud CLI
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Option B: Use service account key
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/your/service-account-key.json"
```

### Step 2: Configure Environment

```bash
# Edit .env file with your real project details
GOOGLE_CLOUD_PROJECT=your-actual-project-id
GOOGLE_APPLICATION_CREDENTIALS=/path/to/your/service-account.json
```

### Step 3: Populate with Real Data

```bash
# This will fetch real data from your GCP project
python populate_sqlite.py
```

### Step 4: Start Services

```bash
./scripts/start_services.sh
```

---

## 🛠️ Troubleshooting

### Backend Won't Start
- Check if port 8000 is already in use: `lsof -i :8000`
- Check logs: `tail -f backend_logs.txt`

### Frontend Won't Start
- Check if port 8501 is already in use: `lsof -i :8501`
- Check logs: `tail -f frontend_logs.txt`

### Agent Returns Raw JSON Instead of Analysis
- Make sure you're using the latest code with session management fixes
- Check backend logs for ADK initialization errors
- Verify the database has data: `ls -la backend/cache/gcp_data.db`

### No Data in Database
- Run demo data setup: `python scripts/setup_demo_data.py`
- Check database exists: `ls -la backend/cache/`

### Import Errors
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check if you're in the right directory and virtual environment

---

## 📊 Test Suite

Run the test suite to verify everything works:

```bash
# Test basic functionality
python test_agent.py

# Test LLM analysis quality
python test_enhanced_agent_analysis.py

# Test response quality assessment
python tests/test_response_quality.py
```

---

## 🔧 Development Mode

For development, you can monitor logs in real-time:

```bash
# Monitor backend logs
tail -f backend_logs.txt

# Monitor frontend logs
tail -f frontend_logs.txt

# Monitor both
./scripts/monitor_logs.sh
```

---

## 📁 Project Structure

```
security_agent/
├── backend/               # FastAPI backend
├── frontend/             # Streamlit frontend
├── agents/               # ADK agent and tools
├── scripts/              # Setup and utility scripts
├── tests/                # Test files
├── docs/                 # Documentation
├── .env.example          # Environment template
├── run_backend.py        # Backend startup
├── run_frontend.py       # Frontend startup
└── requirements.txt      # Python dependencies
```

---

## 🎯 What's Different About This Agent

This security agent provides **LLM-powered analysis** instead of just returning raw data:

- **Before**: `{"success": true, "data": [...14 buckets...]}`
- **After**: *"Based on analysis of your 14 storage buckets, I recommend prioritizing these security concerns: 1. Critical Risk: Three buckets lack encryption..."*

The agent uses the Google ADK (Agent Development Kit) with Gemini 2.5 Flash to provide intelligent, contextual security insights.