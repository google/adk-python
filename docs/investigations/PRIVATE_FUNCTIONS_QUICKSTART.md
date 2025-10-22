# Private Cloud Functions - Quick Start Guide

**Last Updated:** 2025-10-22
**Branch:** `claude/investigate-cloud-functions-011CUNbxHLneQM4LpjHRkVWH`

## Overview

This guide provides a quick reference for migrating Cloud Functions to private, internal-only access.

## Current vs. Target Architecture

| Aspect | Current (Public) | Target (Private) |
|--------|-----------------|------------------|
| **Access** | Public internet | Internal only |
| **Authentication** | None (--allow-unauthenticated) | OIDC token required |
| **Network** | No VPC connectivity | Direct VPC egress |
| **Service Account** | Default Compute SA | Dedicated SA with least privilege |
| **Ingress** | Allow all | Internal only |
| **Security** | ⚠️ Low | ✅ High |

## Quick Migration Steps

### ⚠️ CRITICAL: Check Organization Policies First

Before attempting migration, verify your organization's policies allow private Cloud Functions:

```bash
# Check organization policy compliance
./scripts/deployment/check_org_policies.sh
```

**If compliance check fails**, see [`ORGANIZATION_POLICY_COMPLIANCE.md`](./ORGANIZATION_POLICY_COMPLIANCE.md) for resolution steps.

### Option 1: Automated Migration (Recommended)

```bash
# Step 0: Verify organization policy compliance
./scripts/deployment/check_org_policies.sh

# Step 1: Setup VPC infrastructure
./scripts/deployment/setup_vpc_infrastructure.sh

# Step 2: (Optional) Setup VPC Service Controls for maximum security
./scripts/deployment/setup_vpc_service_controls.sh

# Step 3: Deploy private cloud functions
./scripts/deployment/deploy_private_cloud_functions.sh

# Step 4: Test authentication and connectivity
./scripts/testing/test_private_functions.sh

# Step 5: (If VPC-SC enabled) Test VPC Service Controls compliance
./scripts/testing/test_vpc_service_controls.sh
```

### Option 2: Manual Migration

```bash
# 1. Enable required APIs
gcloud services enable vpcaccess.googleapis.com compute.googleapis.com

# 2. Create service account
gcloud iam service-accounts create security-fetcher-sa \
  --display-name="Security Fetcher Function"

# 3. Grant IAM roles
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

# 4. Deploy with private settings
gcloud functions deploy unified-security-fetcher \
  --gen2 \
  --region=us-central1 \
  --ingress-settings=internal-only \
  --no-allow-unauthenticated \
  --service-account=security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --vpc-egress=private-ranges-only \
  --network=projects/${PROJECT_ID}/global/networks/default \
  --subnet=projects/${PROJECT_ID}/regions/us-central1/subnetworks/default

# 5. Update Cloud Scheduler with OIDC auth
gcloud scheduler jobs update http unified-security-findings-trigger \
  --oidc-service-account-email=${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
  --oidc-token-audience="https://us-central1-${PROJECT_ID}.cloudfunctions.net/unified-security-fetcher"
```

## Key Configuration Changes

### Deployment Script Changes

**Before:**
```bash
gcloud functions deploy unified-security-fetcher \
  --trigger-http \
  --allow-unauthenticated
```

**After:**
```bash
gcloud functions deploy unified-security-fetcher \
  --ingress-settings=internal-only \
  --no-allow-unauthenticated \
  --service-account=security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com \
  --vpc-egress=private-ranges-only \
  --network=projects/${PROJECT_ID}/global/networks/default \
  --subnet=projects/${PROJECT_ID}/regions/us-central1/subnetworks/default
```

### Cloud Scheduler Authentication

**Before:**
```bash
gcloud scheduler jobs create http job-name \
  --uri="https://..." \
  --http-method=GET
```

**After:**
```bash
gcloud scheduler jobs create http job-name \
  --uri="https://..." \
  --http-method=GET \
  --oidc-service-account-email=${SA_EMAIL} \
  --oidc-token-audience=${FUNCTION_URL}
```

### Programmatic Invocation

**Before (Python):**
```python
import requests

response = requests.post(
    "https://us-central1-project.cloudfunctions.net/function",
    json={"key": "value"}
)
```

**After (Python):**
```python
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token
import requests

# Get ID token
auth_req = google.auth.transport.requests.Request()
target_audience = "https://us-central1-project.cloudfunctions.net/function"
token = id_token.fetch_id_token(auth_req, target_audience)

# Make authenticated request
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    target_audience,
    json={"key": "value"},
    headers=headers
)
```

## Verification Checklist

### Pre-Deployment
- [ ] Organization policies checked and compliant
- [ ] VPC infrastructure created
- [ ] Service accounts created with minimal permissions
- [ ] Required APIs enabled

### Post-Deployment
- [ ] Public access returns 403 Forbidden
- [ ] Authenticated access returns 200 OK
- [ ] Cloud Scheduler jobs trigger successfully
- [ ] Function can access BigQuery
- [ ] Function can access Security Command Center
- [ ] IAM permissions are minimal (least privilege)
- [ ] VPC egress is configured
- [ ] Service account is dedicated (not default)

## Troubleshooting

### Issue: "Permission denied" (403)

**Cause:** Missing IAM roles

**Solution:**
```bash
# Grant Cloud Functions Invoker role
gcloud functions add-iam-policy-binding unified-security-fetcher \
  --region=us-central1 \
  --member="serviceAccount:YOUR_SA@project.iam.gserviceaccount.com" \
  --role="roles/cloudfunctions.invoker"
```

### Issue: "Function not found" (404)

**Cause:** Function not deployed or wrong region

**Solution:**
```bash
# List functions
gcloud functions list --region=us-central1

# Check deployment status
gcloud functions describe unified-security-fetcher --region=us-central1
```

### Issue: Cloud Scheduler fails to trigger

**Cause:** Missing OIDC configuration

**Solution:**
```bash
# Update scheduler job with OIDC auth
./scripts/deployment/deploy_private_cloud_functions.sh
```

### Issue: Function can't access BigQuery

**Cause:** Service account missing BigQuery roles

**Solution:**
```bash
gcloud projects add-iam-policy-binding $PROJECT_ID \
  --member="serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"
```

## Cost Impact

| Resource | Current | After Migration | Difference |
|----------|---------|----------------|------------|
| Cloud Function | $10-50/mo | $10-50/mo | No change |
| VPC Connector | $0 | $0 (using Direct VPC) | No change |
| Cloud NAT | $0 | $0-30/mo (optional) | +$0-30/mo |
| **Total** | **$10-50/mo** | **$10-80/mo** | **+$0-30/mo** |

**Note:** Using Direct VPC egress (recommended) has no additional cost vs. VPC Connector ($40-200/mo).

## Performance Impact

| Metric | Current | After Migration | Change |
|--------|---------|----------------|--------|
| Cold start | ~2-3s | ~2-3s | No change |
| Warm latency | ~100ms | ~120ms | +20ms (VPC overhead) |
| Throughput | 100 req/s | 100 req/s | No change |

## Security Improvements

| Control | Before | After (Private) | After (+ VPC-SC) |
|---------|--------|----------------|------------------|
| Public access | ❌ Yes | ✅ No | ✅ No |
| Authentication | ❌ None | ✅ OIDC tokens | ✅ OIDC tokens |
| Network isolation | ❌ No | ✅ VPC isolated | ✅ VPC isolated |
| Service account | ⚠️ Default (Editor) | ✅ Dedicated | ✅ Dedicated |
| Audit logging | ⚠️ Partial | ✅ Complete | ✅ Complete |
| **Data exfiltration prevention** | ❌ No | ⚠️ Limited | ✅ Org-level perimeter |
| **Cross-project access control** | ❌ No | ⚠️ IAM only | ✅ Perimeter enforced |
| **Compliance** | ⚠️ Basic | ✅ Good | ✅ HIPAA/PCI-DSS/SOC2 |

## Next Steps

1. **Review:** Share migration plan with team
2. **Test:** Deploy to test environment first
3. **Monitor:** Set up alerting for authentication failures
4. **Update:** Modify Security Agent to use authenticated calls
5. **Document:** Update runbooks and incident response procedures

## Files Created

| File | Purpose |
|------|---------|
| `docs/investigations/private-cloud-functions-migration.md` | Comprehensive migration guide |
| `docs/investigations/PRIVATE_FUNCTIONS_QUICKSTART.md` | This quick start guide |
| `docs/investigations/ORGANIZATION_POLICY_COMPLIANCE.md` | Organization policy compliance assessment |
| `docs/investigations/VPC_SERVICE_CONTROLS_GUIDE.md` | VPC Service Controls comprehensive guide |
| `scripts/deployment/check_org_policies.sh` | Organization policy compliance checker |
| `scripts/deployment/setup_vpc_infrastructure.sh` | VPC setup automation |
| `scripts/deployment/setup_vpc_service_controls.sh` | VPC Service Controls setup automation |
| `scripts/deployment/deploy_private_cloud_functions.sh` | Private function deployment |
| `scripts/testing/test_private_functions.sh` | Validation test suite |
| `scripts/testing/test_vpc_service_controls.sh` | VPC-SC compliance test suite |

## Support

For questions or issues:
1. Check the comprehensive migration guide: `docs/investigations/private-cloud-functions-migration.md`
2. Review GCP documentation: https://cloud.google.com/functions/docs/networking
3. Contact the security team for organization policy questions

---

**Remember:** Always test in a non-production environment first!
