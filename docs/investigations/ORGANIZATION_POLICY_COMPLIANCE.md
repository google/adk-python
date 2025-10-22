# Organization Policy Compliance Assessment
# Private Cloud Functions Migration

**Date:** 2025-10-22
**Branch:** `claude/investigate-cloud-functions-011CUNbxHLneQM4LpjHRkVWH`
**Status:** Compliance Analysis Complete

## Executive Summary

This document assesses GCP Organization Policy constraints that may affect or block the private Cloud Functions migration. Understanding these policies is **critical** before attempting deployment, as violations will cause deployment failures.

**Key Finding:** ✅ The proposed solution is **designed to be compliant** with common security-focused organization policies. However, you must verify your organization's specific policies before proceeding.

---

## Critical Organization Policies for Cloud Functions Gen2

### 1. `constraints/run.allowedIngress` ⭐ CRITICAL

**What it controls:** Restricts which ingress settings developers can use for Cloud Run (Cloud Functions Gen2)

**Possible values:**
- `all` - Allow all traffic (least secure)
- `internal` - Internal traffic only (recommended)
- `internal-and-cloud-load-balancing` - Internal + external load balancer

**Our solution compliance:**
- ✅ **Uses:** `--ingress-settings=internal-only`
- ✅ **Compliant with:** `internal` or `internal-and-cloud-load-balancing` policies
- ❌ **Blocked if policy requires:** N/A (our solution is most restrictive)

**How to check:**
```bash
gcloud resource-manager org-policies describe \
  run.allowedIngress \
  --project=$PROJECT_ID \
  --effective
```

**If policy blocks "internal":**
- This is unlikely, as "internal" is the most secure setting
- If blocked, you may need to use `internal-and-cloud-load-balancing`
- Contact your GCP org admin for policy exception

---

### 2. `constraints/run.allowedVPCEgress` ⭐ CRITICAL

**What it controls:** Restricts VPC egress settings for Cloud Run services

**Possible values:**
- `all-traffic` - Route all traffic through VPC (most restrictive)
- `private-ranges-only` - Route only RFC1918 ranges through VPC (recommended)

**Our solution compliance:**
- ✅ **Uses:** `--vpc-egress=private-ranges-only`
- ✅ **Compliant with:** Both `all-traffic` and `private-ranges-only` policies

**How to check:**
```bash
gcloud resource-manager org-policies describe \
  run.allowedVPCEgress \
  --project=$PROJECT_ID \
  --effective
```

**If policy requires "all-traffic":**
- Update deployment script to use `--vpc-egress=all-traffic`
- Note: This routes ALL traffic (including Google APIs) through VPC
- May require Cloud NAT for internet/API access

---

### 3. `constraints/cloudfunctions.requireVPCConnector` 🔶 IMPORTANT

**What it controls:** Requires all Cloud Functions to use a VPC connector

**Our solution compliance:**
- ⚠️ **Uses:** Direct VPC egress (recommended, more performant)
- ❌ **Not compliant if:** Policy enforces VPC connector requirement
- ✅ **Alternative:** Use VPC connector instead of Direct VPC egress

**How to check:**
```bash
gcloud resource-manager org-policies describe \
  cloudfunctions.requireVPCConnector \
  --project=$PROJECT_ID \
  --effective
```

**If policy requires VPC connector:**
1. Create VPC connector: `./scripts/deployment/setup_vpc_infrastructure.sh` (answer "yes")
2. Update deployment to use connector:
```bash
gcloud functions deploy unified-security-fetcher \
  --vpc-connector=security-agent-connector \
  --vpc-egress=private-ranges-only \
  # Remove: --network and --subnet flags
```

**Cost impact:** +$40-200/month for VPC connector instances

---

### 4. `constraints/iam.allowedPolicyMemberDomains` 🔶 IMPORTANT

**What it controls:** Restricts which domains can be granted IAM roles

**Possible values:**
- `C0xxxxxxx` - Your organization's customer ID
- Specific domains like `example.com`

**Our solution compliance:**
- ✅ **Uses:** Service accounts within the same project
- ✅ **Compliant:** Service accounts are in the same organization

**How to check:**
```bash
gcloud resource-manager org-policies describe \
  iam.allowedPolicyMemberDomains \
  --project=$PROJECT_ID \
  --effective
```

**If policy restricts domains:**
- Ensure service accounts are created in the same organization
- Our deployment script creates SAs in the same project (compliant)

---

### 5. `constraints/compute.trustedImageProjects` 🔶 IMPORTANT

**What it controls:** Restricts which projects can provide VM images (affects VPC connector)

**Required for:** VPC connector creation (uses Deployment Manager VMs)

**Our solution compliance:**
- ⚠️ **Required only if using VPC connector**
- ✅ **Direct VPC egress:** Not affected by this policy

**How to check:**
```bash
gcloud resource-manager org-policies describe \
  compute.trustedImageProjects \
  --project=$PROJECT_ID \
  --effective
```

**If using VPC connector and policy is enforced:**
- Must allow project: `serverless-vpc-access-images`
- Contact org admin to add exception:
```bash
gcloud resource-manager org-policies set-policy policy.yaml --project=$PROJECT_ID
```

**policy.yaml:**
```yaml
name: projects/$PROJECT_ID/policies/compute.trustedImageProjects
spec:
  rules:
    - allowAll: false
      values:
        allowedValues:
          - projects/serverless-vpc-access-images
```

---

### 6. `constraints/compute.vmExternalIpAccess` 🔷 MODERATE

**What it controls:** Restricts external IPs on Compute Engine VMs

**Our solution compliance:**
- ✅ **Cloud Functions Gen2:** Does not use Compute Engine VMs directly
- ⚠️ **VPC Connector:** May be affected if policy is very restrictive

**How to check:**
```bash
gcloud resource-manager org-policies describe \
  compute.vmExternalIpAccess \
  --project=$PROJECT_ID \
  --effective
```

---

### 7. `constraints/compute.requireShieldedVm` 🔷 MODERATE

**What it controls:** Requires VMs to use Shielded VM features

**Our solution compliance:**
- ✅ **Cloud Functions Gen2:** Automatically uses secure execution environments
- ⚠️ **VPC Connector:** Deployment Manager may need Shielded VM support

---

### 8. VPC Service Controls ⭐ CRITICAL

**What it controls:** Creates security perimeters around GCP services

**Our solution compliance:**
- ✅ **Compatible:** Cloud Functions Gen2 supports VPC Service Controls
- ⚠️ **Configuration required:** Must add services to perimeter

**Services to include in perimeter:**
- `cloudfunctions.googleapis.com`
- `run.googleapis.com`
- `bigquery.googleapis.com`
- `securitycenter.googleapis.com`
- `compute.googleapis.com`
- `storage.googleapis.com`
- `iam.googleapis.com`

**How to check:**
```bash
# List all service perimeters
gcloud access-context-manager perimeters list \
  --policy=$(gcloud access-context-manager policies list --format="value(name)")

# Describe specific perimeter
gcloud access-context-manager perimeters describe PERIMETER_NAME \
  --policy=POLICY_NAME
```

**If project is in a VPC-SC perimeter:**
1. Ensure all required services are in `restrictedServices`
2. Configure ingress/egress rules for Cloud Scheduler
3. Add Cloud Functions to allowed services
4. Test in non-production first

**Example perimeter configuration:**
```bash
gcloud access-context-manager perimeters update PERIMETER_NAME \
  --add-restricted-services=cloudfunctions.googleapis.com,run.googleapis.com \
  --policy=POLICY_NAME
```

---

## Compliance Verification Checklist

Use this checklist before attempting migration:

### Pre-Migration Checks

- [ ] Run organization policy check script: `./scripts/deployment/check_org_policies.sh`
- [ ] Verify `run.allowedIngress` allows "internal" setting
- [ ] Verify `run.allowedVPCEgress` allows "private-ranges-only"
- [ ] Check if `cloudfunctions.requireVPCConnector` is enforced
- [ ] Verify IAM domain restrictions don't block service accounts
- [ ] Check for VPC Service Controls perimeters
- [ ] Confirm project is not in VPC-SC perimeter OR perimeter allows Cloud Functions
- [ ] Verify trusted image projects include `serverless-vpc-access-images` (if using VPC connector)

### Deployment Checks

- [ ] Test deployment in non-production project first
- [ ] Monitor for organization policy violations in deployment logs
- [ ] Verify function deploys successfully
- [ ] Test authenticated access works
- [ ] Validate Cloud Scheduler can invoke function
- [ ] Check Cloud Audit Logs for policy violations

### Post-Deployment Checks

- [ ] Audit function configuration matches organization policies
- [ ] Verify no policy drift over time
- [ ] Set up monitoring for policy changes
- [ ] Document any policy exceptions granted

---

## Common Policy Conflict Scenarios

### Scenario 1: VPC Connector Required

**Symptom:** Deployment fails with "VPC connector required by organization policy"

**Solution:**
```bash
# Create VPC connector
./scripts/deployment/setup_vpc_infrastructure.sh
# Answer "yes" when prompted

# Update deployment script
# Change from:
--vpc-egress=private-ranges-only \
--network=projects/${PROJECT_ID}/global/networks/default \
--subnet=projects/${PROJECT_ID}/regions/us-central1/subnetworks/default

# To:
--vpc-connector=security-agent-connector \
--vpc-egress=private-ranges-only
```

**Cost impact:** +$40-200/month

---

### Scenario 2: Internal Ingress Not Allowed

**Symptom:** Deployment fails with "Ingress setting not allowed by organization policy"

**Solution:**
1. Check what's allowed:
```bash
gcloud resource-manager org-policies describe run.allowedIngress \
  --project=$PROJECT_ID --effective
```

2. Update deployment to use allowed value (e.g., `internal-and-cloud-load-balancing`)

3. If policy requires "all", contact org admin for exception (security risk)

---

### Scenario 3: VPC-SC Perimeter Blocks Deployment

**Symptom:** Deployment succeeds but function can't access BigQuery/SCC

**Solution:**
1. Add required services to VPC-SC perimeter restricted services
2. Configure ingress rules to allow Cloud Build
3. Configure egress rules for BigQuery, Security Command Center
4. Test thoroughly in non-production

---

### Scenario 4: Trusted Image Project Missing

**Symptom:** VPC connector creation fails with "Image not allowed"

**Solution:**
```bash
# Add serverless-vpc-access-images to trusted projects
cat > trusted-images-policy.yaml <<EOF
name: projects/$PROJECT_ID/policies/compute.trustedImageProjects
spec:
  rules:
    - allowAll: false
      values:
        allowedValues:
          - projects/serverless-vpc-access-images
          - projects/$PROJECT_ID
EOF

gcloud resource-manager org-policies set-policy trusted-images-policy.yaml
```

---

## Automated Compliance Checking

Created script: `scripts/deployment/check_org_policies.sh`

**Usage:**
```bash
./scripts/deployment/check_org_policies.sh
```

**Output:**
- ✅ Compliant policies
- ⚠️ Warning: May require attention
- ❌ Blocking: Will prevent deployment

---

## Requesting Policy Exceptions

If your organization policies block the migration, follow this process:

### 1. Document Business Justification

**Template:**
```
Request: Exception for Cloud Functions private deployment

Justification:
- Migrating Cloud Functions to private, internal-only access
- Improves security posture by blocking public internet access
- Requires [specific policy change]
- Affects project: [PROJECT_ID]
- Duration: Permanent for production security

Security benefits:
- Removes public attack surface
- Enforces authentication for all invocations
- Network-level isolation via VPC
- Principle of least privilege service accounts

Alternative considered:
- [List alternatives and why they don't work]
```

### 2. Submit Exception Request

Contact your:
- GCP Organization Admin
- Cloud Security Team
- Compliance Team

### 3. Implement with Least Privilege

If exception granted:
- Apply policy at project level (not org level) if possible
- Document exception in security review
- Set up monitoring for policy compliance
- Plan for regular review (quarterly)

---

## Monitoring & Compliance Drift

### Set Up Alerting

**Policy change notifications:**
```bash
# Create Pub/Sub topic for policy changes
gcloud pubsub topics create org-policy-changes

# Create log sink for policy modifications
gcloud logging sinks create org-policy-sink \
  pubsub.googleapis.com/projects/$PROJECT_ID/topics/org-policy-changes \
  --log-filter='protoPayload.methodName="SetOrgPolicy" OR protoPayload.methodName="DeleteOrgPolicy"'
```

### Regular Audits

**Monthly:**
- Review organization policy changes
- Audit Cloud Functions for compliance
- Check for policy drift

**Quarterly:**
- Full security review
- Update documentation
- Test exception validity

**Annually:**
- Comprehensive compliance assessment
- Review policy exceptions
- Update security controls

---

## Recommendations

### For Organizations WITHOUT Restrictive Policies

✅ **Proceed with Direct VPC Egress deployment:**
- Most cost-effective ($0 additional)
- Best performance
- Simplest architecture
- Run: `./scripts/deployment/deploy_private_cloud_functions.sh`

### For Organizations WITH VPC Connector Requirement

⚠️ **Use VPC Connector deployment:**
- Additional cost: $40-200/month
- Slightly higher latency
- Run: `./scripts/deployment/setup_vpc_infrastructure.sh` (create connector)
- Modify deployment script to use connector

### For Organizations WITH VPC Service Controls

⚠️ **Extra configuration required:**
1. Work with org admin to add services to perimeter
2. Configure ingress/egress rules
3. Test extensively in non-production
4. Budget extra time for troubleshooting (1-2 weeks)

### For Organizations WITH Strict Image Policies

⚠️ **Add serverless-vpc-access-images to trusted projects:**
- Required for VPC connector creation
- Request exception from org admin
- Alternative: Use Direct VPC egress (no VM images needed)

---

## Summary Matrix

| Organization Policy | Our Solution | Compliance Status | Action Required |
|-------------------|-------------|------------------|-----------------|
| `run.allowedIngress` | `internal-only` | ✅ Compliant | None - most restrictive setting |
| `run.allowedVPCEgress` | `private-ranges-only` | ✅ Compliant | None - standard setting |
| `cloudfunctions.requireVPCConnector` | Direct VPC egress | ⚠️ May conflict | Check policy; use connector if required |
| `iam.allowedPolicyMemberDomains` | Same-org SAs | ✅ Compliant | None - service accounts in same org |
| `compute.trustedImageProjects` | Not applicable* | ✅ Compliant | None if using Direct VPC; exception needed for connector |
| VPC Service Controls | Supported | ✅ Compatible | Configure perimeter if active |

*Direct VPC egress doesn't use VM images; only VPC connector requires this

---

## Next Steps

1. **Run compliance check:** `./scripts/deployment/check_org_policies.sh`
2. **Review results** with your team and org admin
3. **Request exceptions** if needed (2-4 weeks lead time)
4. **Test in non-production** once policies are verified
5. **Proceed with migration** following phased rollout plan

---

## Support Resources

- **GCP Documentation:** [Organization Policy Constraints](https://cloud.google.com/resource-manager/docs/organization-policy/org-policy-constraints)
- **Cloud Run Constraints:** [Custom Constraints](https://cloud.google.com/run/docs/securing/custom-constraints)
- **VPC Service Controls:** [Using VPC-SC](https://cloud.google.com/run/docs/securing/using-vpc-service-controls)
- **Internal:** Contact your GCP organization administrator

---

**Last Updated:** 2025-10-22
**Maintained By:** Security & Infrastructure Team
**Review Frequency:** Quarterly or when policies change
