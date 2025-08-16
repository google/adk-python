# RADAR Agent Architecture for Cloud Operations

## Overview
RADAR is an operations-focused pattern for Day 2 cloud management:
- **R**ecognize: Discovery and inventory of existing resources
- **A**ssess: Security, compliance, and health evaluation  
- **D**ecide: Prioritization and recommendation generation
- **A**ct: Remediation and configuration changes
- **R**eview: Verification, monitoring, and continuous improvement

## Agent Specialization and Tool Assignment

### 1. Recognition Agent
**Purpose**: Discovers what exists in your cloud environment
**Primary Tools**:
- `list_gcp_assets` - Complete inventory
- `search_gcp_assets` - Find specific resources
- `list_service_accounts` - Identity inventory

**Why these tools**: This agent needs read-only discovery capabilities. It's the "eyes" of the system, cataloging everything that exists.

### 2. Assessment Agent  
**Purpose**: Evaluates security posture and compliance
**Primary Tools**:
- `list_security_findings` - Active threats
- `analyze_iam_security` - Permission risks
- `analyze_api_keys_security` - Key vulnerabilities
- `get_security_stats` - Trend analysis
- `comprehensive_security_scan` - Full evaluation

**Why these tools**: This agent needs deep analysis capabilities across multiple security domains. It interprets what the Recognition Agent found.

### 3. Decision Agent
**Purpose**: Prioritizes issues and generates recommendations
**Primary Tools**:
- `analyze_advisory_notifications` - External threat intelligence
- `list_advisory_notifications` - Security bulletins
- `get_security_stats` - For prioritization
- `analyze_iam_security` - For recommendation generation

**Tool Overlap Rationale**: This agent needs some of the same analysis tools as Assessment Agent but uses them differently - for prioritization rather than discovery.

### 4. Action Agent
**Purpose**: Executes remediation and changes
**Primary Tools**:
- `create_api_key` - With proper restrictions
- `get_iam_policy` - Before modifications
- `get_notification_settings` - Configuration management

**Why these tools**: This agent has write capabilities. It's intentionally limited to prevent unauthorized changes.

### 5. Review Agent
**Purpose**: Verifies changes and reports status
**Primary Tools**:
- `comprehensive_security_scan` - Post-change verification
- `list_security_findings` - Confirm remediation
- `get_security_stats` - Track improvements
- `list_gcp_assets` - Verify inventory changes

**Tool Overlap Rationale**: Needs the same observation tools as Assessment Agent but uses them to verify Action Agent's work.

## Intentional Tool Overlap

### Why Multiple Agents Have `analyze_iam_security`:
- **Assessment Agent**: Uses it to identify current risks
- **Decision Agent**: Uses it to generate specific recommendations
- **Review Agent**: Uses it to verify improvements

### Why Multiple Agents Have `list_security_findings`:
- **Assessment Agent**: Initial security posture snapshot
- **Review Agent**: Confirms findings were addressed
- **Decision Agent**: Prioritizes which findings to address first

## Agent Collaboration Pattern

```
User Query → Recognition Agent (What do we have?)
                ↓
           Assessment Agent (What's wrong?)
                ↓
           Decision Agent (What should we fix first?)
                ↓
           Action Agent (Execute fixes)
                ↓
           Review Agent (Did it work?)
                ↓
           Report back to user
```

## Example Scenarios

### Scenario 1: "Check our security posture"
1. **Recognition Agent**: Inventories all resources
2. **Assessment Agent**: Runs security scans on discovered resources
3. **Decision Agent**: Prioritizes critical findings
4. **Review Agent**: Generates executive report

### Scenario 2: "Fix critical API key issues"
1. **Assessment Agent**: Identifies unrestricted keys
2. **Decision Agent**: Determines which keys are most critical
3. **Action Agent**: Applies restrictions to keys
4. **Review Agent**: Verifies restrictions are in place

### Scenario 3: "What changed since last week?"
1. **Recognition Agent**: Current inventory
2. **Review Agent**: Compares with historical data
3. **Assessment Agent**: Evaluates security impact of changes
4. **Decision Agent**: Recommends actions for new resources