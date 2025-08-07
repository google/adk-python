# Environment Configuration Guide

The Security Agent now supports comprehensive environment configuration through a `.env` file. This makes deployment and development much more flexible and secure.

## 📁 Directory Structure

**IMPORTANT:** Always run `python run.py` from the **root directory** of the security_agent project:

```
security_agent/                    ← RUN FROM HERE
├── .env                          ← Environment configuration
├── .env.example                  ← Template file
├── run.py                        ← Main startup script
├── backend/                      ← Backend code
├── frontend/                     ← Frontend code
├── venv/                         ← Virtual environment
└── logs/                         ← Log files
```

## Quick Setup

1. **Navigate to project root:**
   ```bash
   cd /path/to/security_agent      # Make sure you're in the root directory
   ```

2. **Copy the example file:**
   ```bash
   cp .env.example .env
   ```

3. **Edit your `.env` file with your settings:**
   ```bash
   nano .env  # or use your preferred editor
   ```

4. **Run the application from root:**
   ```bash
   python run.py  # Automatically loads .env from root
   ```

## Environment Variables

### Google Cloud Configuration
```env
# Required for GCP services
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_APPLICATION_CREDENTIALS=backend/config/secrets/service-account-key.json
```

### Service Configuration
```env
# Choose backend architecture
USE_MODULAR=true                          # Use modular backend (recommended)
# USE_MODULAR=false                       # Use legacy backend

# Service configuration file
SERVICE_CONFIG_PATH=backend/config/services.json
```

### Server Configuration
```env
# Backend server settings
PORT=8000                                 # Backend port
HOST=0.0.0.0                             # Backend host
LOG_LEVEL=info                           # Logging level (debug, info, warning, error)
RELOAD=true                              # Auto-reload on file changes (dev only)

# Frontend server settings
FRONTEND_PORT=8501                       # Streamlit frontend port
FRONTEND_HOST=0.0.0.0                   # Frontend host
```

### Production Settings
```env
PRODUCTION=false                         # Production mode
DEBUG=true                               # Debug mode
WORKERS=1                                # Number of uvicorn workers (production)
```

### Logging Configuration
```env
LOG_TO_FILE=true                         # Enable file logging
LOG_FILE_PATH=logs/app.log              # Log file location
LOG_MAX_SIZE=10485760                    # Max log file size (10MB)
LOG_BACKUP_COUNT=5                       # Number of backup log files
```

### Security Settings
```env
SECRET_KEY=your-secret-key-here         # Application secret key
ALLOWED_HOSTS=localhost,127.0.0.1,0.0.0.0  # Allowed hosts
```

### Feature Flags
```env
# Enable/disable specific services
IAM_ANALYSIS_ENABLED=true
COMPLIANCE_ENABLED=true
THREAT_INTELLIGENCE_ENABLED=false
SECURITY_ANALYTICS_ENABLED=false

# System features
ENABLE_TRACING=false
ENABLE_METRICS=false
ENABLE_PROFILING=false
```

## 🚀 Running the Application

### Always Run From Root Directory

```bash
# ✅ CORRECT - Run from security_agent root
cd /Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent
python run.py

# ❌ WRONG - Don't run from subdirectories  
cd backend
python ../run.py              # This won't work properly

# ❌ WRONG - Don't run from frontend
cd frontend  
python ../run.py              # This won't work properly
```

### Verify You're in the Right Directory

```bash
# You should see these files in your current directory:
ls -la
# Should show: .env, run.py, backend/, frontend/, venv/, etc.

# Current working directory should end with 'security_agent'
pwd
# Should show: .../security_agent
```

## Usage Examples

### Local Development
```env
# .env for development
USE_MODULAR=true
PORT=8000
FRONTEND_PORT=8501
RELOAD=true
DEBUG=true
LOG_LEVEL=debug
GOOGLE_CLOUD_PROJECT=my-dev-project
```

### Production Deployment
```env
# .env for production
USE_MODULAR=true
PORT=8000
RELOAD=false
DEBUG=false
PRODUCTION=true
LOG_LEVEL=info
WORKERS=4
GOOGLE_CLOUD_PROJECT=my-prod-project
```

### Cloud Run Deployment
```env
# .env for Cloud Run (these are often set by Cloud Run automatically)
USE_MODULAR=true
PORT=8080                               # Cloud Run sets this
K_SERVICE=security-agent
GOOGLE_CLOUD_PROJECT=my-project
```

## Command Line Override

Environment variables can be overridden by command line arguments:

```bash
# Override USE_MODULAR from .env
python run.py --modular              # Forces modular mode
python run.py                        # Uses .env setting

# Other overrides work similarly
python run.py --production           # Forces production mode
```

## Deployment Modes

### 1. Local Development
```bash
# Uses .env file automatically
python run.py
```

### 2. Cloud Mode
```bash
# For running in containers/Cloud Run
python run.py --cloud
```

### 3. Direct Deployment
```bash
# Deploy directly to Cloud Run
python run.py --deploy PROJECT_ID
```

### 4. Docker
```bash
# Docker automatically includes .env
python run.py --docker
```

## Environment Validation

The application will:
- ✅ Automatically detect and load `.env` files
- ✅ Validate required environment variables
- ✅ Provide helpful error messages for missing config
- ✅ Support both relative and absolute paths
- ✅ Auto-install `python-dotenv` if needed

## Security Best Practices

1. **Never commit `.env` to version control**
   ```bash
   # .gitignore already includes
   .env
   .env.*
   ```

2. **Use different `.env` files for different environments**
   ```bash
   cp .env.example .env.development
   cp .env.example .env.production
   ```

3. **Store secrets securely**
   - Use Google Secret Manager for production
   - Use service account keys with minimal permissions
   - Rotate credentials regularly

4. **Validate sensitive paths**
   - Service account keys should have restricted file permissions
   - Use absolute paths in production

## Troubleshooting

### Common Issues

**`.env` file not loading:**
- Check file exists in project root
- Ensure `python-dotenv` is installed: `pip install python-dotenv`
- Check file permissions

**GCP authentication fails:**
- Verify `GOOGLE_APPLICATION_CREDENTIALS` path is correct
- Check service account key file exists and has proper permissions
- Ensure `GOOGLE_CLOUD_PROJECT` is set correctly

**Service fails to start:**
- Check port conflicts (`PORT`, `FRONTEND_PORT`)
- Verify all required environment variables are set
- Check log files for detailed error messages

### Debug Mode

Enable debug logging to troubleshoot:
```env
LOG_LEVEL=debug
DEBUG=true
```

Then check logs:
```bash
tail -f logs/app.log
```