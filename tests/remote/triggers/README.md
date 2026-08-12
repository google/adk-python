# Remote Integration Tests for `/apps/{app_name}/trigger/*` Endpoints

End-to-end tests that verify Pub/Sub push, and
Eventarc triggers against a real ADK agent deployed to Cloud Run.

## Workflow

This workflow is split into three independent phases.

### Phase 1: Deploy Agent

Build the ADK package and deploy the echo agent to Cloud Run. This ensures that
the service and its identity exist before any infrastructure wiring occurs.

```bash
export GCP_PROJECT_ID=your-project-id
export SUFFIX=ea786 # Pick a unique suffix
export SERVICE_NAME=adk-trigger-test-$SUFFIX

# 1. Build local ADK wheel
uv build --wheel --out-dir tests/remote/triggers/test_agent/wheels/

# 2. Deploy Agent
gcloud run deploy $SERVICE_NAME \
  --source=tests/remote/triggers/test_agent \
  --project="$GCP_PROJECT_ID" \
  --region="us-central1" \
  --port=8080 \
  --quiet
```

### Phase 2: Wire Infrastructure (Terraform)

Run Terraform to create the supporting infrastructure (IAM roles, Pub/Sub
topics).

```bash
cd tests/remote/triggers/terraform
terraform init
terraform apply \
  -var=project_id=$GCP_PROJECT_ID \
  -var=service_name=$SERVICE_NAME \
  -var=suffix=$SUFFIX
```

### Phase 3: Run Tests (Pytest)

```bash
# Run Tests from the project root
export GCP_PROJECT_ID=your-project-id
export SUFFIX=ea786
pytest tests/remote/triggers/ -v -s
```

## Cleanup

```bash
# 1. Destroy infrastructure
cd tests/remote/triggers/terraform
terraform destroy \
  -var=project_id=$GCP_PROJECT_ID \
  -var=service_name=$SERVICE_NAME \
  -var=suffix=$SUFFIX

# 2. Delete Cloud Run service
gcloud run services delete $SERVICE_NAME --project=$GCP_PROJECT_ID --region=us-central1 --quiet
```
