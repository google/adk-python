# GCP Service Account Credentials

This directory is for storing your Google Cloud Platform service account JSON key file.

## Setup Instructions

1. **Create a GCP Service Account**
   ```bash
   # Set your project ID
   export PROJECT_ID=your-project-id

   # Create service account
   gcloud iam service-accounts create security-agent-sa \
     --display-name="Security Agent Service Account" \
     --project=$PROJECT_ID
   ```

2. **Grant Required Permissions**
   ```bash
   # Required: BigQuery access
   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
     --role="roles/bigquery.dataViewer"

   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
     --role="roles/bigquery.jobUser"

   gcloud projects add-iam-policy-binding $PROJECT_ID \
     --member="serviceAccount:security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
     --role="roles/aiplatform.user"
   ```

3. **Download JSON Key**
   ```bash
   gcloud iam service-accounts keys create service-account-key.json \
     --iam-account=security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com

   # Move key to this directory
   mv service-account-key.json config/

   # Secure the file
   chmod 600 config/service-account-key.json
   ```

4. **Update .env File**
   ```bash
   # Ensure your .env file points to this key
   GOOGLE_APPLICATION_CREDENTIALS=config/service-account-key.json
   ```

## Security Notes

- **Never commit JSON keys to git** - This directory has a `.gitignore` to prevent accidental commits
- **Keep permissions strict** - Use `chmod 600` on JSON files
- **Rotate keys regularly** - Delete old keys after creating new ones
- **Use least privilege** - Only grant the minimum required roles

## File Naming Convention

The default expected filename is: `service-account-key.json`

You can use a different name, but update the `.env` file accordingly:
```bash
GOOGLE_APPLICATION_CREDENTIALS=config/your-custom-name.json
```

## Verification

Test your credentials are working:
```bash
# Export credentials
export GOOGLE_APPLICATION_CREDENTIALS=config/service-account-key.json

# Test authentication
gcloud auth activate-service-account --key-file=$GOOGLE_APPLICATION_CREDENTIALS

# Test BigQuery access
bq ls --project_id=$GOOGLE_CLOUD_PROJECT
```
