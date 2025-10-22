# Private Cloud Functions Migration Investigation

**Date:** 2025-10-22
**Branch:** `claude/investigate-cloud-functions-011CUNbxHLneQM4LpjHRkVWH`
**Status:** Investigation Complete - Ready for Implementation Planning

## Executive Summary

This document outlines the investigation and migration strategy for converting the current public Cloud Functions deployment to a private, internal-only architecture where all traffic is restricted to internal Google Cloud networks.

**Current State:**
- Unified Cloud Function deployed with `--allow-unauthenticated`
- HTTP triggers accessible from public internet
- No VPC connectivity configured
- Cloud Scheduler triggers via public URLs

**Target State:**
- Private Cloud Functions with internal-only ingress
- VPC connectivity for internal traffic routing
- Service account-based authentication
- VPC Service Controls perimeter protection

---

## Current Architecture Analysis

### Deployment Configuration

**Function:** `unified-security-fetcher`
- **Location:** `us-central1`
- **Runtime:** `python311`
- **Generation:** Gen2
- **Memory:** 1024MB
- **Timeout:** 540s (9 minutes)
- **Max Instances:** 10
- **Trigger:** HTTP (`--trigger-http`)
- **Authentication:** None (`--allow-unauthenticated`)

### Network Configuration

**Current Ingress Settings:**
```bash
# Default: Allow all traffic
--trigger-http --allow-unauthenticated
```

**Exposed Endpoints:**
```
POST /fetch/{fetcher_name}      # Public HTTP endpoint
POST /fetch/all                 # Public HTTP endpoint
GET  /fetchers                  # Public HTTP endpoint
GET  /health                    # Public HTTP endpoint
GET  /docs                      # Public API documentation
GET  /trigger/{fetcher_name}    # Cloud Scheduler endpoint
```

**Callers:**
- Cloud Scheduler (9 jobs triggering various fetchers)
- Security Agent (queries BigQuery data)
- Potential: Internal services, Cloud Run, other Cloud Functions

### Security Posture

**Current Vulnerabilities:**
1. ⚠️ Functions accessible from public internet
2. ⚠️ No authentication/authorization required
3. ⚠️ Function URLs can be discovered and called by anyone
4. ⚠️ No network-level isolation
5. ⚠️ Using default Compute Engine service account (overly permissive)

---

## Private Cloud Functions Requirements

### 1. VPC Connectivity Options

#### Option A: Direct VPC Egress (Recommended)

**Advantages:**
- ✅ No connector infrastructure needed
- ✅ Pay only for network traffic (scales to zero)
- ✅ Use network tags directly on service revisions
- ✅ Lower latency and higher throughput
- ✅ No compute charges for connector instances

**Use Cases:**
- Accessing resources in VPC (databases, internal APIs)
- Routing traffic through VPC firewall rules
- Service-to-service communication within same project

**Configuration:**
```bash
gcloud functions deploy unified-security-fetcher \
  --gen2 \
  --vpc-connector="" \
  --vpc-egress=private-ranges-only \
  --network=projects/${PROJECT_ID}/global/networks/default \
  --subnet=projects/${PROJECT_ID}/regions/us-central1/subnetworks/default
```

#### Option B: Serverless VPC Access Connector

**When to Use:**
- Need connectivity to on-premises networks via VPN
- Require specific IP ranges for firewall rules
- Need cross-project VPC connectivity

**Configuration:**
1. Create VPC Connector:
```bash
gcloud compute networks vpc-access connectors create security-agent-connector \
  --region=us-central1 \
  --network=default \
  --range=10.8.0.0/28 \
  --min-instances=2 \
  --max-instances=10
```

2. Deploy with connector:
```bash
gcloud functions deploy unified-security-fetcher \
  --gen2 \
  --vpc-connector=security-agent-connector \
  --vpc-egress=private-ranges-only
```

**Costs:**
- Connector runs 2-10 instances continuously
- Charged for compute time even at minimum instances
- ~$40-200/month depending on scale

### 2. Ingress Settings

#### Internal-Only Configuration

**Option:** `--ingress-settings=internal-only`

**Allowed Traffic Sources:**
- ✅ Cloud Scheduler
- ✅ Cloud Tasks
- ✅ Eventarc
- ✅ Workflows
- ✅ BigQuery
- ✅ VPC networks in same project
- ✅ Resources in same VPC Service Controls perimeter
- ❌ Public internet requests

**Deployment Command:**
```bash
gcloud functions deploy unified-security-fetcher \
  --gen2 \
  --ingress-settings=internal-only \
  --no-allow-unauthenticated \
  --service-account=security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

### 3. Authentication & Authorization

#### Service Account Configuration

**Step 1: Create Dedicated Service Account**
```bash
# Create function service account
gcloud iam service-accounts create security-fetcher-sa \
  --display-name="Security Fetcher Function Service Account" \
  --project=${PROJECT_ID}

# Grant necessary permissions (principle of least privilege)
gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/bigquery.dataEditor"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/securitycenter.findingsEditor"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/compute.viewer"

gcloud projects add-iam-policy-binding ${PROJECT_ID} \
  --member="serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/iam.securityReviewer"
```

**Step 2: Grant Cloud Functions Invoker Role**
```bash
# Allow Cloud Scheduler to invoke function
gcloud functions add-iam-policy-binding unified-security-fetcher \
  --region=us-central1 \
  --member="serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com" \
  --role="roles/cloudfunctions.invoker"

# Allow other services to invoke (if needed)
gcloud functions add-iam-policy-binding unified-security-fetcher \
  --region=us-central1 \
  --member="serviceAccount:security-agent-sa@${PROJECT_ID}.iam.gserviceaccount.com" \
  --role="roles/cloudfunctions.invoker"
```

#### Service-to-Service Authentication

**For Cloud Scheduler:**
```bash
gcloud scheduler jobs create http security-findings-job \
  --location=us-central1 \
  --schedule="0 */2 * * *" \
  --uri="https://us-central1-${PROJECT_ID}.cloudfunctions.net/unified-security-fetcher/trigger/security_findings" \
  --http-method=GET \
  --oidc-service-account-email=${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
  --oidc-token-audience="https://us-central1-${PROJECT_ID}.cloudfunctions.net/unified-security-fetcher"
```

**For programmatic invocation:**
```python
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token

# Get credentials and ID token
auth_req = google.auth.transport.requests.Request()
target_audience = "https://us-central1-PROJECT_ID.cloudfunctions.net/unified-security-fetcher"
token = id_token.fetch_id_token(auth_req, target_audience)

# Make authenticated request
headers = {"Authorization": f"Bearer {token}"}
response = requests.post(
    f"{target_audience}/fetch/security_findings",
    headers=headers
)
```

### 4. VPC Service Controls

#### Creating a Service Perimeter

**Step 1: Define Perimeter**
```bash
# Create access policy (if not exists)
gcloud access-context-manager policies create \
  --title="Security Agent Policy" \
  --organization=${ORG_ID}

# Get policy name
POLICY_NAME=$(gcloud access-context-manager policies list --format="value(name)")

# Create service perimeter
gcloud access-context-manager perimeters create security_agent_perimeter \
  --title="Security Agent Perimeter" \
  --policy=${POLICY_NAME} \
  --resources=projects/${PROJECT_NUMBER} \
  --restricted-services=cloudfunctions.googleapis.com,bigquery.googleapis.com,securitycenter.googleapis.com \
  --enable-vpc-accessible-services \
  --vpc-allowed-services=cloudfunctions.googleapis.com,bigquery.googleapis.com,securitycenter.googleapis.com
```

**Step 2: Organization Policy**
```bash
# Enforce internal-only ingress
cat > ingress-policy.yaml <<EOF
name: projects/${PROJECT_ID}/policies/run.allowedIngress
spec:
  rules:
    - values:
        allowedValues:
          - "internal"
EOF

gcloud resource-manager org-policies set-policy ingress-policy.yaml \
  --project=${PROJECT_ID}
```

---

## Migration Strategy

### Phase 1: Preparation (Week 1)

**Infrastructure Setup:**

1. **Create VPC Resources**
   - [ ] Create or identify VPC network
   - [ ] Create subnet in us-central1 with appropriate CIDR
   - [ ] (Optional) Create VPC Connector if not using Direct VPC egress
   - [ ] Configure firewall rules for internal traffic

2. **Service Account Creation**
   - [ ] Create dedicated service account for Cloud Function
   - [ ] Grant minimum required IAM roles
   - [ ] Create service account for invoking clients
   - [ ] Grant `roles/cloudfunctions.invoker` to authorized services

3. **Testing Environment**
   - [ ] Set up test project or isolated environment
   - [ ] Deploy test version of function with private settings
   - [ ] Validate all integrations work with authentication

**Validation:**
```bash
# Test direct database access
python test_adk_query.py

# Test API endpoint authentication
./scripts/test_private_function.sh
```

### Phase 2: Deployment Configuration Update (Week 1-2)

**Update Deployment Script** (`cloud_functions/unified/deploy.sh`)

**Changes:**
```bash
# Add VPC configuration
VPC_NETWORK="default"
VPC_SUBNET="default"
SERVICE_ACCOUNT="security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com"

# Update gcloud deploy command
gcloud functions deploy unified-security-fetcher \
  --gen2 \
  --region=us-central1 \
  --runtime=python311 \
  --source=. \
  --entry-point=unified_handler \
  --trigger-http \
  --memory=1024Mi \
  --timeout=540s \
  --max-instances=10 \
  --ingress-settings=internal-only \
  --no-allow-unauthenticated \
  --service-account=${SERVICE_ACCOUNT} \
  --vpc-egress=private-ranges-only \
  --network=projects/${PROJECT_ID}/global/networks/${VPC_NETWORK} \
  --subnet=projects/${PROJECT_ID}/regions/us-central1/subnetworks/${VPC_SUBNET} \
  --set-env-vars=PROJECT_ID=${PROJECT_ID},BQ_DATASET_ID=security_insights,BQ_LOCATION=us-central1
```

**Update Cloud Build Config** (`cloud_functions/unified/cloudbuild.yaml`)

```yaml
steps:
  - name: 'gcr.io/google.com/cloudsdktool/cloud-sdk'
    entrypoint: 'bash'
    args:
      - '-c'
      - |
        gcloud functions deploy unified-security-fetcher \
          --gen2 \
          --region=us-central1 \
          --runtime=python311 \
          --source=cloud_functions/unified \
          --entry-point=unified_handler \
          --trigger-http \
          --ingress-settings=internal-only \
          --no-allow-unauthenticated \
          --service-account=${_SERVICE_ACCOUNT} \
          --vpc-egress=private-ranges-only \
          --network=projects/${PROJECT_ID}/global/networks/default \
          --subnet=projects/${PROJECT_ID}/regions/us-central1/subnetworks/default \
          --memory=1024Mi \
          --timeout=540s \
          --max-instances=10

substitutions:
  _SERVICE_ACCOUNT: security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com
```

### Phase 3: Cloud Scheduler Update (Week 2)

**Update All Scheduler Jobs with OIDC Authentication:**

```bash
# Example: Update security-findings job
gcloud scheduler jobs update http unified-security-findings-trigger \
  --location=us-central1 \
  --schedule="0 */2 * * *" \
  --uri="https://us-central1-${PROJECT_ID}.cloudfunctions.net/unified-security-fetcher/trigger/security_findings" \
  --http-method=GET \
  --oidc-service-account-email=${PROJECT_NUMBER}-compute@developer.gserviceaccount.com \
  --oidc-token-audience="https://us-central1-${PROJECT_ID}.cloudfunctions.net/unified-security-fetcher"
```

**Create helper script:** `scripts/update_scheduler_jobs_private.sh`

```bash
#!/bin/bash
set -e

PROJECT_ID=$(gcloud config get-value project)
PROJECT_NUMBER=$(gcloud projects describe ${PROJECT_ID} --format="value(projectNumber)")
FUNCTION_URL="https://us-central1-${PROJECT_ID}.cloudfunctions.net/unified-security-fetcher"
SCHEDULER_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"

# Array of fetcher names and schedules
declare -A SCHEDULES
SCHEDULES=(
  ["security_findings"]="0 */2 * * *"
  ["custom_roles"]="0 9 * * *"
  ["compute_instances"]="0 */4 * * *"
  ["firewall_rules"]="0 */4 * * *"
  ["storage_buckets"]="0 */6 * * *"
  ["iam_accounts"]="0 */4 * * *"
  ["service_account_roles"]="0 */4 * * *"
  ["standard_roles"]="0 9 * * 1"
  ["user_roles"]="0 */4 * * *"
)

for FETCHER in "${!SCHEDULES[@]}"; do
  JOB_NAME="unified-${FETCHER}-trigger"
  SCHEDULE="${SCHEDULES[$FETCHER]}"

  echo "Updating scheduler job: ${JOB_NAME}"

  gcloud scheduler jobs update http ${JOB_NAME} \
    --location=us-central1 \
    --schedule="${SCHEDULE}" \
    --uri="${FUNCTION_URL}/trigger/${FETCHER}" \
    --http-method=GET \
    --oidc-service-account-email=${SCHEDULER_SA} \
    --oidc-token-audience=${FUNCTION_URL} \
    --time-zone="America/New_York"
done

echo "All scheduler jobs updated with OIDC authentication!"
```

### Phase 4: Security Agent Integration (Week 2-3)

**Update Agent Authentication** (`agents/security_agent.py`)

**Add authentication for function invocation:**

```python
import google.auth
import google.auth.transport.requests
from google.oauth2 import id_token

class SecurityAgent:
    def __init__(self):
        self.function_url = os.getenv(
            "CLOUD_FUNCTION_URL",
            f"https://us-central1-{project_id}.cloudfunctions.net/unified-security-fetcher"
        )

    def get_auth_token(self) -> str:
        """Get ID token for authenticating to Cloud Function."""
        auth_req = google.auth.transport.requests.Request()
        return id_token.fetch_id_token(auth_req, self.function_url)

    def trigger_fetcher(self, fetcher_name: str) -> dict:
        """Trigger a specific fetcher with authentication."""
        token = self.get_auth_token()
        headers = {"Authorization": f"Bearer {token}"}

        response = requests.post(
            f"{self.function_url}/fetch/{fetcher_name}",
            headers=headers
        )
        response.raise_for_status()
        return response.json()
```

### Phase 5: Testing & Validation (Week 3)

**Test Checklist:**

1. **Internal Access Validation**
   - [ ] Cloud Scheduler can trigger functions
   - [ ] Security Agent can invoke functions
   - [ ] Functions can access BigQuery
   - [ ] Functions can access Security Command Center
   - [ ] Functions can access IAM/Compute/Storage APIs

2. **External Access Blocking**
   - [ ] Public HTTP requests are rejected (401/403)
   - [ ] Unauthenticated requests are rejected
   - [ ] Invalid tokens are rejected
   - [ ] Cross-project access without perimeter fails

3. **Performance Testing**
   - [ ] Latency within acceptable range (<500ms overhead)
   - [ ] Throughput meets requirements
   - [ ] Concurrent requests handled properly
   - [ ] VPC egress bandwidth sufficient

4. **Error Handling**
   - [ ] Authentication failures return proper errors
   - [ ] Network failures are logged
   - [ ] Retry logic works for transient failures
   - [ ] Audit logs capture all access attempts

**Test Scripts:**

**Test 1: Public Access Blocked**
```bash
# Should fail with 403 Forbidden
curl -X POST https://us-central1-${PROJECT_ID}.cloudfunctions.net/unified-security-fetcher/fetch/security_findings
```

**Test 2: Authenticated Access Works**
```bash
# Should succeed
./scripts/test_authenticated_function.sh
```

```bash
#!/bin/bash
# scripts/test_authenticated_function.sh

PROJECT_ID=$(gcloud config get-value project)
FUNCTION_URL="https://us-central1-${PROJECT_ID}.cloudfunctions.net/unified-security-fetcher"

# Get ID token
TOKEN=$(gcloud auth print-identity-token --audiences=${FUNCTION_URL})

# Test health endpoint
echo "Testing health endpoint..."
curl -H "Authorization: Bearer ${TOKEN}" ${FUNCTION_URL}/health

# Test fetcher trigger
echo "Testing security_findings fetcher..."
curl -H "Authorization: Bearer ${TOKEN}" -X POST ${FUNCTION_URL}/fetch/security_findings
```

**Test 3: Cloud Scheduler Integration**
```bash
# Manually trigger scheduler job
gcloud scheduler jobs run unified-security-findings-trigger --location=us-central1

# Check logs
gcloud functions logs read unified-security-fetcher --region=us-central1 --limit=50
```

### Phase 6: Production Rollout (Week 4)

**Gradual Rollout Strategy:**

1. **Deploy to Test Environment**
   - Deploy with private settings in test project
   - Run full test suite
   - Monitor for 48 hours

2. **Deploy to Staging**
   - Deploy to production project with private settings
   - Keep old function as backup
   - Route 10% traffic to new function
   - Monitor for 24 hours

3. **Full Production**
   - Route 100% traffic to private function
   - Monitor for 48 hours
   - Delete old public function

**Rollback Plan:**
```bash
# If issues occur, redeploy with public settings
gcloud functions deploy unified-security-fetcher \
  --gen2 \
  --trigger-http \
  --allow-unauthenticated \
  --ingress-settings=all
```

---

## Infrastructure Requirements

### 1. VPC Network Configuration

**Resources Needed:**

| Resource | Configuration | Cost Estimate |
|----------|--------------|---------------|
| VPC Network | Use existing `default` network | Free |
| Subnet | `10.128.0.0/20` in us-central1 | Free |
| VPC Connector (optional) | 2-10 instances, /28 range | $40-200/month |
| Firewall Rules | Allow internal traffic | Free |

**Firewall Rules:**

```bash
# Allow internal traffic from Cloud Functions
gcloud compute firewall-rules create allow-cloud-functions-internal \
  --network=default \
  --allow=tcp,udp,icmp \
  --source-ranges=10.128.0.0/20 \
  --description="Allow traffic from Cloud Functions via VPC"
```

### 2. IAM Configuration

**Service Accounts:**

| Service Account | Purpose | Roles Required |
|----------------|---------|----------------|
| `security-fetcher-sa` | Cloud Function runtime | `bigquery.dataEditor`<br>`securitycenter.findingsEditor`<br>`compute.viewer`<br>`iam.securityReviewer` |
| Cloud Scheduler Default | Invoke function | `cloudfunctions.invoker` |
| Security Agent SA | Query BigQuery | `bigquery.dataViewer` |

### 3. VPC Service Controls (Optional - High Security)

**Perimeter Configuration:**

```bash
# Protected services
- cloudfunctions.googleapis.com
- bigquery.googleapis.com
- securitycenter.googleapis.com
- compute.googleapis.com
- storage.googleapis.com
- iam.googleapis.com

# Access levels
- Device policy: Require Corp device
- IP restrictions: Allow only corporate IP ranges
- Service account restrictions: Only authorized SAs
```

---

## Cost Analysis

### Current Architecture Costs

| Component | Cost |
|-----------|------|
| Cloud Function | $0.40/million invocations + $0.0000025/GB-sec |
| Cloud Scheduler | $0.10/job/month (9 jobs = $0.90) |
| BigQuery | Storage + queries (variable) |
| **Total** | ~$10-50/month (depends on usage) |

### Private Architecture Additional Costs

| Component | Cost | Notes |
|-----------|------|-------|
| Direct VPC Egress | Network egress only | Recommended, minimal overhead |
| VPC Connector (if used) | $40-200/month | 2-10 instances |
| VPC Service Controls | Free | No additional cost |
| **Additional Cost** | $0-200/month | Depends on VPC strategy |

**Recommendation:** Use Direct VPC Egress to minimize costs while maintaining security.

---

## Security Benefits

### Before Migration

| Risk | Severity | Impact |
|------|----------|--------|
| Public internet exposure | High | Functions can be discovered and called by anyone |
| No authentication | High | Data can be accessed without authorization |
| Default SA permissions | Medium | Overly broad permissions if compromised |
| No network isolation | Medium | Function traffic not isolated |

### After Migration

| Control | Benefit |
|---------|---------|
| Internal-only ingress | ✅ No public internet access |
| Service account auth | ✅ Only authorized services can invoke |
| Dedicated SA | ✅ Principle of least privilege |
| VPC isolation | ✅ Network-level security |
| VPC Service Controls | ✅ Organization-level perimeter |
| Audit logging | ✅ Complete access trail |

---

## Monitoring & Observability

### Key Metrics to Track

**Function Performance:**
```bash
# Cloud Monitoring queries
# Invocation count by status
resource.type="cloud_function"
resource.labels.function_name="unified-security-fetcher"
metric.type="cloudfunctions.googleapis.com/function/execution_count"

# Execution time
metric.type="cloudfunctions.googleapis.com/function/execution_times"

# Network egress
metric.type="cloudfunctions.googleapis.com/function/network_egress"
```

**Security Metrics:**
```bash
# Failed authentication attempts
resource.type="cloud_function"
protoPayload.status.code!=0
protoPayload.status.message=~"authentication"

# Unauthorized access attempts
resource.type="cloud_function"
httpRequest.status=403 OR httpRequest.status=401
```

**Alerting Policies:**

1. **High authentication failure rate**
   - Threshold: >10 failures/5 minutes
   - Action: Alert security team

2. **Function invocation errors**
   - Threshold: Error rate >5%
   - Action: Alert on-call engineer

3. **VPC connector saturation** (if using connector)
   - Threshold: >80% utilization
   - Action: Scale up connector

---

## Risk Assessment & Mitigation

### Migration Risks

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| Cloud Scheduler auth breaks | Medium | High | Test in staging, keep backup function |
| Security Agent can't invoke | Medium | High | Update agent with auth token logic |
| VPC connectivity issues | Low | High | Use Direct VPC egress (simpler) |
| Increased latency | Low | Medium | Monitor performance, tune VPC config |
| Cost increase | Low | Low | Use Direct VPC egress to minimize |

### Rollback Plan

**If critical issues occur:**

1. Redeploy function with public settings (5 minutes)
2. Revert Cloud Scheduler jobs to no-auth (5 minutes)
3. Update agent to remove auth logic (code rollback)
4. Investigate issues in test environment

**Recovery Time Objective (RTO):** 15 minutes
**Recovery Point Objective (RPO):** 0 (no data loss)

---

## Recommendations

### Immediate Actions (Week 1)

1. **Use Direct VPC Egress** - Simpler, cheaper, more performant than VPC Connector
2. **Create Dedicated Service Account** - Principle of least privilege
3. **Enable Cloud Audit Logs** - Track all access attempts
4. **Test in Staging First** - Validate all integrations before production

### Short-term (Weeks 2-4)

1. **Deploy with `--ingress-settings=internal-only`**
2. **Update Cloud Scheduler with OIDC authentication**
3. **Update Security Agent with auth token logic**
4. **Monitor for 2 weeks before considering production-ready**

### Long-term (Months 2-3)

1. **Implement VPC Service Controls** - Organization-level perimeter
2. **Add mutual TLS** - Additional layer of authentication
3. **Implement rate limiting** - Protect against abuse
4. **Add Cloud Armor** - DDoS protection and WAF rules

### Optional Enhancements

1. **Private Service Connect** - For cross-project connectivity
2. **Cloud NAT** - For outbound internet access from VPC
3. **VPN/Interconnect** - For on-premises connectivity
4. **Workload Identity** - For GKE integration

---

## Next Steps

1. **Review this document** with stakeholders
2. **Approve migration plan** and timeline
3. **Allocate engineering resources** (1-2 weeks of work)
4. **Set up test environment** in separate project
5. **Execute Phase 1** (infrastructure setup)
6. **Begin implementation** following phased approach

---

## References

- [GCP Cloud Functions Networking](https://cloud.google.com/functions/docs/networking/network-settings)
- [Direct VPC Egress](https://cloud.google.com/vpc/docs/configure-serverless-vpc-access)
- [VPC Service Controls](https://cloud.google.com/run/docs/securing/using-vpc-service-controls)
- [Private Cloud Functions](https://cloud.google.com/run/docs/securing/private-networking)
- [Service Account Authentication](https://cloud.google.com/functions/docs/securing/function-identity)
- [Cloud Scheduler Authentication](https://cloud.google.com/scheduler/docs/http-target-auth)

---

## Appendix A: Complete Deployment Script

See: `scripts/deployment/deploy_private_cloud_functions.sh`

## Appendix B: Test Scripts

See: `scripts/testing/test_private_functions.sh`

## Appendix C: Monitoring Dashboard

See: `monitoring/private-functions-dashboard.json`
