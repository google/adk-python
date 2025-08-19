# Enhanced IAM Security Analysis API Documentation

## Overview

The Enhanced IAM Security Analysis API provides comprehensive IAM security assessment including overprivileged account detection, service account key rotation analysis, and automated risk scoring. This API is part of STORY-003: IAM Assessment Enhancement.

**Base URL**: `/api/v1/iam`

## Core Endpoint

### Enhanced IAM Security Analysis

Perform comprehensive IAM security analysis with advanced detection algorithms.

**Endpoint**: `GET /api/v1/iam/analyze`

**Query Parameters**:
- `project_id` (optional): GCP project ID to analyze (defaults to configured project)

**Response**:
```json
{
  "success": true,
  "source": "enhanced_iam_analyzer",
  "analysis": {
    "project_id": "my-project-id",
    "posture_score": 72,
    "risk_distribution": {
      "CRITICAL": 1,
      "HIGH": 2,
      "MEDIUM": 3,
      "LOW": 1,
      "MINIMAL": 0
    },
    "total_findings": 7,
    "critical_findings": 1,
    "high_findings": 2,
    "statistics": {
      "service_account_count": 15,
      "overprivileged_accounts": 3,
      "stale_keys": 5,
      "cross_project_bindings": 1,
      "external_users": 2
    },
    "recommendations": [
      "🔴 CRITICAL: Address wildcard IAM bindings and admin role misuse immediately",
      "🟠 HIGH: Review 3 overprivileged service accounts and implement least privilege",
      "🟡 MEDIUM: Rotate 5 stale service account keys (>90 days old)",
      "Implement automated service account key rotation",
      "Use Workload Identity for GKE workloads instead of service account keys"
    ],
    "findings": [
      {
        "type": "ADMIN_ROLE_MISUSE",
        "risk_level": "CRITICAL",
        "risk_score": 90,
        "title": "Service Account with Owner Role",
        "description": "Service account has admin roles: roles/owner",
        "resource_name": "projects/my-project/serviceAccounts/admin-sa@my-project.iam.gserviceaccount.com",
        "affected_principal": "admin-sa@my-project.iam.gserviceaccount.com",
        "remediation_steps": [
          "Review if admin role is truly necessary",
          "Consider using more specific roles",
          "Implement least privilege principle",
          "Use Workload Identity if possible"
        ],
        "metadata": {
          "admin_roles": ["roles/owner"]
        },
        "detected_at": "2024-01-15T10:30:00Z"
      }
    ],
    "analyzed_at": "2024-01-15T10:30:00Z"
  }
}
```

**Status Codes**:
- `200 OK`: Analysis completed successfully
- `500 Internal Server Error`: Analysis failed

---

## Finding Types

The enhanced analyzer detects the following security issues:

### ADMIN_ROLE_MISUSE
- **Risk Level**: CRITICAL (90-95 points)
- **Description**: Service accounts with admin roles (owner, iam.securityAdmin, etc.)
- **Impact**: High privilege escalation risk
- **Remediation**: Replace with specific roles, implement least privilege

### OVERPRIVILEGED_SERVICE_ACCOUNT
- **Risk Level**: HIGH (70-80 points)
- **Description**: Service accounts with multiple high-privilege roles
- **Impact**: Excessive permissions beyond requirements
- **Remediation**: Consolidate roles, remove unnecessary permissions

### EXCESSIVE_PERMISSIONS
- **Risk Level**: HIGH (60-70 points)
- **Description**: Service accounts with broad roles (editor, admin)
- **Impact**: Over-broad access to resources
- **Remediation**: Replace with granular, specific roles

### STALE_SERVICE_ACCOUNT_KEY
- **Risk Level**: MEDIUM-HIGH (50-80 points based on age)
- **Description**: Service account keys older than 90 days
- **Impact**: Increased compromise risk, compliance violations
- **Remediation**: Rotate keys, implement automated rotation

### WILDCARD_BINDING
- **Risk Level**: CRITICAL (95 points)
- **Description**: Roles granted to allUsers or allAuthenticatedUsers
- **Impact**: Public access to resources
- **Remediation**: Remove wildcard bindings, grant specific access

### EXTERNAL_USER_ACCESS
- **Risk Level**: MEDIUM (60 points)
- **Description**: Users from external domains with project access
- **Impact**: Potential unauthorized access
- **Remediation**: Verify necessity, use guest accounts

### UNUSED_SERVICE_ACCOUNT
- **Risk Level**: LOW (20 points)
- **Description**: Service accounts with no keys that may be unused
- **Impact**: Attack surface expansion
- **Remediation**: Delete if unused, document purpose

### CROSS_PROJECT_ACCESS
- **Risk Level**: MEDIUM (50 points)
- **Description**: Service accounts with access to other projects
- **Impact**: Lateral movement risk
- **Remediation**: Review necessity, implement project isolation

---

## Risk Scoring Algorithm

### Posture Score Calculation
The overall security posture score (0-100) is calculated using weighted penalties:

```
Posture Score = 100 - (
  CRITICAL_FINDINGS × 25 +
  HIGH_FINDINGS × 15 +
  MEDIUM_FINDINGS × 8 +
  LOW_FINDINGS × 3 +
  MINIMAL_FINDINGS × 1
)
```

### Individual Finding Risk Scores
- **CRITICAL**: 85-95 points
- **HIGH**: 65-85 points
- **MEDIUM**: 40-65 points
- **LOW**: 15-40 points
- **MINIMAL**: 5-15 points

### Risk Level Thresholds
- **🔴 CRITICAL (85-100)**: Immediate action required
- **🟠 HIGH (65-84)**: Address within 1 week
- **🟡 MEDIUM (40-64)**: Address within 1 month
- **🔵 LOW (15-39)**: Address during next review cycle
- **🟢 MINIMAL (0-14)**: Monitor and document

---

## Enhanced Features

### Overprivileged Account Detection
- **Admin Role Detection**: Identifies service accounts with owner, security admin, or IAM admin roles
- **Broad Role Analysis**: Detects editor, compute admin, storage admin assignments
- **Multiple High-Privilege Roles**: Flags accounts with 3+ high-privilege roles
- **Custom Role Analysis**: Evaluates custom roles for excessive permissions

### Service Account Key Analysis
- **Age Tracking**: Identifies keys older than 90 days (configurable threshold)
- **Usage Patterns**: Detects unused service accounts (no keys, not disabled)
- **Key Algorithm Check**: Identifies weak or deprecated key algorithms
- **Rotation Recommendations**: Provides automated rotation guidance

### Cross-Project Security
- **Cross-Project Bindings**: Detects service accounts with multi-project access
- **Organization-Level Analysis**: Identifies org-wide permissions
- **Resource Hierarchy Review**: Analyzes folder and project-level access

### External Access Detection
- **Domain Analysis**: Identifies users from external domains
- **Guest Account Detection**: Flags non-organizational users
- **Wildcard Binding Alert**: Critical alerts for allUsers/allAuthenticatedUsers

---

## Integration Examples

### ADK Agent Commands
```
"Analyze my IAM security posture"
"Show me overprivileged service accounts"
"Check for stale service account keys"
"What are my critical IAM findings?"
"Generate IAM security recommendations"
```

### Programmatic Usage
```python
import httpx

async def analyze_iam_security(project_id=None):
    async with httpx.AsyncClient() as client:
        params = {"project_id": project_id} if project_id else {}
        response = await client.get(
            "http://localhost:8000/api/v1/iam/analyze",
            params=params
        )
        return response.json()

# Usage
analysis = await analyze_iam_security("my-project")
posture_score = analysis["analysis"]["posture_score"]
critical_findings = analysis["analysis"]["critical_findings"]
```

---

## Automated Recommendations

The system generates prioritized, actionable recommendations:

### Priority 1: Critical Issues
- Address wildcard IAM bindings immediately
- Remove admin roles from service accounts
- Implement emergency access procedures

### Priority 2: High-Risk Issues
- Review and reduce overprivileged accounts
- Implement least privilege access
- Enable IAM conditions for temporary access

### Priority 3: Medium-Risk Issues
- Rotate stale service account keys
- Implement automated key rotation
- Review external user access

### Priority 4: Best Practices
- Use Workload Identity instead of service account keys
- Enable IAM audit logging
- Conduct quarterly access reviews
- Create custom roles for specific permissions

---

## Compliance Mapping

### Security Frameworks
- **NIST Cybersecurity Framework**: AC-2, AC-3, AC-6
- **CIS Controls**: Control 5 (Account Management), Control 6 (Access Control)
- **ISO 27001**: A.9.2 (User Access Management)
- **SOC 2**: CC6.1, CC6.2, CC6.3

### GCP Security Best Practices
- Principle of least privilege
- Regular access reviews
- Automated key rotation
- Workload Identity adoption
- IAM conditions for fine-grained access

---

## Performance and Limits

### API Performance
- **Response Time**: < 5 seconds for typical projects
- **Scalability**: Supports projects with 1000+ service accounts
- **Caching**: Results cached for 5 minutes

### Rate Limits
- **Analysis Requests**: 10 per minute per project
- **Heavy Analysis**: 2 per minute for projects with 500+ service accounts

### Data Retention
- **Analysis Results**: Stored for 30 days
- **Finding History**: Maintained for trend analysis
- **Audit Trail**: All analyses logged for compliance

---

## Error Handling

### Common Error Scenarios
```json
{
  "success": false,
  "error": "Insufficient permissions to analyze IAM",
  "details": {
    "required_permissions": [
      "iam.serviceAccounts.list",
      "resourcemanager.projects.getIamPolicy"
    ]
  }
}
```

### Required Permissions
- `iam.serviceAccounts.list`
- `iam.serviceAccountKeys.list`
- `resourcemanager.projects.getIamPolicy`
- `cloudasset.assets.searchAllResources` (optional, for enhanced analysis)

---

## Security Considerations

- **Credential Management**: Never stores or logs service account keys
- **Access Control**: Requires appropriate IAM permissions
- **Audit Logging**: All analysis requests logged
- **Data Privacy**: No sensitive data transmitted or stored
- **Encryption**: All API communications over HTTPS