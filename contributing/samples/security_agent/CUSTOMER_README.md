# GCP Security Analysis Agent

A comprehensive security analysis platform for Google Cloud Platform (GCP) projects with AI-powered insights and recommendations.

## Features

🔒 **Comprehensive Security Analysis:**
- IAM Policy analysis with security best practices validation
- Google Cloud Active Assist recommendations integration
- Real-time security scoring (0-100 scale)
- Risk assessment and user categorization

📊 **Interactive Dashboard:**
- Visual security posture overview
- Risk distribution charts and gauges  
- Actionable security recommendations
- Direct Google Cloud Console integration

🤖 **AI Security Agent:**
- Conversational security analysis
- Context-aware recommendations
- Quick security insights and guidance

✅ **Compliance & Best Practices:**
- SOC2, ISO27001, GDPR compliance checking
- Security best practices validation
- Risk-based user access reviews

## Quick Start

### Prerequisites

- Python 3.8+
- Google Cloud SDK (`gcloud`) installed and configured
- GCP project with appropriate permissions
- Application Default Credentials configured

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/stuagano/adk-python.git
   cd adk-python/contributing/samples/security_agent
   ```

2. **Run the setup script:**
   ```bash
   ./run.sh
   ```

   This will:
   - Create a Python virtual environment
   - Install all dependencies
   - Start the backend API server
   - Launch the Streamlit frontend
   - Start the ADK web interface (optional)

3. **Access the application:**
   - **Main Dashboard:** http://localhost:8501
   - **API Documentation:** http://localhost:8000/docs
   - **Health Check:** http://localhost:8000/health

### Manual Setup (Alternative)

If you prefer manual setup:

```bash
# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r backend/requirements.txt

# Start backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 &

# Start frontend
cd ..
streamlit run frontend/simple_security_app.py --server.port 8501
```

## Usage

### 1. Security Dashboard
- View overall security posture and score
- Analyze risk distribution across users
- Get quick security insights and recommendations

### 2. IAM Policy Analysis
- Analyze individual user permissions
- Review all users in your project
- Get security recommendations based on best practices

### 3. Active Assist Recommendations
- View Google Cloud security recommendations
- Prioritize actions based on risk level
- Access direct links to Google Cloud Console

### 4. AI Security Agent
- Ask natural language questions about your security posture
- Get personalized recommendations
- Understand complex security concepts

## API Endpoints

The platform provides RESTful APIs for integration:

- `GET /api/v1/gcp/projects` - List available projects
- `GET /api/v1/gcp/project/{project_id}/security-posture` - Overall security summary
- `GET /api/v1/gcp/project/{project_id}/iam/analyze-user/{email}` - User IAM analysis
- `GET /api/v1/gcp/project/{project_id}/iam/analyze-all-users` - Project-wide analysis
- `GET /api/v1/gcp/project/{project_id}/security-recommendations` - Active Assist recommendations

Full API documentation available at http://localhost:8000/docs

## Configuration

### Environment Variables

Create a `.env` file for custom configuration:

```bash
# Google Cloud Project (optional, detected automatically)
GOOGLE_CLOUD_PROJECT=your-project-id

# Backend URL (for frontend)
BACKEND_URL=http://localhost:8000

# Default user email for analysis
DEFAULT_USER_EMAIL=your-email@domain.com
```

### Required GCP Permissions

Your user account needs these IAM roles:
- `roles/browser` - List projects and basic info
- `roles/iam.securityReviewer` - Analyze IAM policies
- `roles/recommender.viewer` - Access Active Assist recommendations

## Troubleshooting

### Common Issues

1. **"Error fetching projects"**
   - Ensure `gcloud auth application-default login` is configured
   - Check that you have `roles/browser` permission

2. **"Backend connection failed"**
   - Verify backend is running on port 8000
   - Check firewall settings

3. **"No recommendations found"**
   - Ensure you have `roles/recommender.viewer` permission
   - Some projects may not have active recommendations

### Getting Help

1. Check the API documentation at http://localhost:8000/docs
2. View logs in the `logs/` directory (created when running)
3. Use the "Raw Data" tab in the frontend for debugging

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │   FastAPI       │    │   Google Cloud  │
│   Frontend      │◄──►│   Backend       │◄──►│   APIs          │
│  (Port 8501)    │    │  (Port 8000)    │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

- **Frontend:** Streamlit-based web interface
- **Backend:** FastAPI with security analysis services
- **APIs:** Google Cloud Resource Manager, IAM, Recommender APIs

## Security Considerations

- This tool analyzes security configurations but does not modify them
- All API calls use your authenticated Google Cloud credentials  
- No sensitive data is stored by the application
- Run in a secure environment and restrict network access as needed

## Contributing

This is a sample application demonstrating Google ADK capabilities. For production use:

1. Review and customize security rules in `backend/services/iam_policy_analyzer.py`
2. Add authentication and authorization as needed
3. Implement proper logging and monitoring
4. Add rate limiting and input validation

## License

This project is part of the Google ADK Python samples and follows the same licensing terms.