# Environment Configuration Reference

## Service Account Configuration

**Service Account File Location:**
```
/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/mgm-digitalconcierge-8ba3b2f28e5f.json
```

**Google Cloud Project:**
```
mgm-digitalconcierge
```

## .env File Settings

The `.env` file is located at:
```
/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/.env
```

### Critical Environment Variables:
```bash
GOOGLE_CLOUD_PROJECT=mgm-digitalconcierge
GOOGLE_APPLICATION_CREDENTIALS=/Users/stuartgano/Desktop/Micron/ADK/contributing/samples/security_agent/mgm-digitalconcierge-8ba3b2f28e5f.json
```

## Important Notes

1. **Service Account Path**: Always use the full absolute path to the service account JSON file
2. **Project ID**: The project ID is `mgm-digitalconcierge`
3. **File Name**: The service account file is `mgm-digitalconcierge-8ba3b2f28e5f.json`

## Troubleshooting

If you see errors like:
- "File backend/config/secrets/your-service-account.json was not found"
- "No module named 'google.adk'"

Check that:
1. The `.env` file has the correct paths (as shown above)
2. The backend has been restarted after updating `.env`
3. The service account file exists at the specified path

## Backend Restart

After updating `.env`, restart the backend:
```bash
# Stop the backend (Ctrl+C) and restart:
python run_backend.py

# Or directly:
cd backend && uvicorn main:app --reload --port 8000
```

---
Last Updated: 2024