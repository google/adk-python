#!/bin/bash
# Check GCP Organization Policies for Cloud Functions Deployment Compliance
# This script validates that the project's organization policies allow private Cloud Functions

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
MAGENTA='\033[0;35m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project 2>/dev/null)}
ORG_ID=$(gcloud projects describe $PROJECT_ID --format="value(parent.id)" 2>/dev/null || echo "")

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Organization Policy Compliance Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Project ID:${NC} $PROJECT_ID"
echo -e "${GREEN}Organization ID:${NC} ${ORG_ID:-"N/A (No organization)"}"
echo ""

COMPLIANCE_SCORE=0
TOTAL_CHECKS=0
WARNINGS=0
ERRORS=0

# Function to check a policy
check_policy() {
  local POLICY_NAME=$1
  local DESCRIPTION=$2
  local REQUIRED_VALUE=$3

  TOTAL_CHECKS=$((TOTAL_CHECKS + 1))

  echo -e "${YELLOW}Checking: ${DESCRIPTION}${NC}"
  echo "  Policy: $POLICY_NAME"

  # Try to get effective policy
  POLICY_OUTPUT=$(gcloud resource-manager org-policies describe "$POLICY_NAME" \
    --project=$PROJECT_ID \
    --effective \
    --format=json 2>/dev/null || echo "{}")

  if [ "$POLICY_OUTPUT" = "{}" ] || [ -z "$POLICY_OUTPUT" ]; then
    echo -e "  ${GREEN}✓ Not enforced${NC} - No restrictions"
    COMPLIANCE_SCORE=$((COMPLIANCE_SCORE + 1))
    echo ""
    return 0
  fi

  # Check if policy is enforced
  IS_BOOLEAN=$(echo "$POLICY_OUTPUT" | jq -r '.booleanPolicy.enforced // false' 2>/dev/null)
  ALLOWED_VALUES=$(echo "$POLICY_OUTPUT" | jq -r '.listPolicy.allowedValues[]? // empty' 2>/dev/null)
  DENIED_VALUES=$(echo "$POLICY_OUTPUT" | jq -r '.listPolicy.deniedValues[]? // empty' 2>/dev/null)

  # Display policy details
  if [ "$IS_BOOLEAN" = "true" ]; then
    echo "  Type: Boolean (Enforced)"
    echo -e "  ${MAGENTA}⚠ WARNING: Policy is enforced${NC}"
    WARNINGS=$((WARNINGS + 1))
  elif [ -n "$ALLOWED_VALUES" ]; then
    echo "  Type: List (Allowed values)"
    echo "  Allowed values:"
    echo "$ALLOWED_VALUES" | while read -r value; do
      echo "    - $value"
    done

    # Check if our required value is allowed
    if [ -n "$REQUIRED_VALUE" ]; then
      if echo "$ALLOWED_VALUES" | grep -q "$REQUIRED_VALUE"; then
        echo -e "  ${GREEN}✓ COMPLIANT${NC} - Required value '$REQUIRED_VALUE' is allowed"
        COMPLIANCE_SCORE=$((COMPLIANCE_SCORE + 1))
      else
        echo -e "  ${RED}✗ BLOCKED${NC} - Required value '$REQUIRED_VALUE' is not in allowed list"
        ERRORS=$((ERRORS + 1))
      fi
    else
      echo -e "  ${GREEN}✓ COMPLIANT${NC} - Policy defined but permissive"
      COMPLIANCE_SCORE=$((COMPLIANCE_SCORE + 1))
    fi
  elif [ -n "$DENIED_VALUES" ]; then
    echo "  Type: List (Denied values)"
    echo "  Denied values:"
    echo "$DENIED_VALUES" | while read -r value; do
      echo "    - $value"
    done

    # Check if our required value is denied
    if [ -n "$REQUIRED_VALUE" ]; then
      if echo "$DENIED_VALUES" | grep -q "$REQUIRED_VALUE"; then
        echo -e "  ${RED}✗ BLOCKED${NC} - Required value '$REQUIRED_VALUE' is denied"
        ERRORS=$((ERRORS + 1))
      else
        echo -e "  ${GREEN}✓ COMPLIANT${NC} - Required value '$REQUIRED_VALUE' is not denied"
        COMPLIANCE_SCORE=$((COMPLIANCE_SCORE + 1))
      fi
    else
      echo -e "  ${GREEN}✓ COMPLIANT${NC} - Policy defined but permissive"
      COMPLIANCE_SCORE=$((COMPLIANCE_SCORE + 1))
    fi
  else
    echo -e "  ${GREEN}✓ Not enforced${NC} - No restrictions"
    COMPLIANCE_SCORE=$((COMPLIANCE_SCORE + 1))
  fi

  echo ""
}

# Check critical policies
echo -e "${BLUE}=== Critical Policies ===${NC}\n"

check_policy "run.allowedIngress" \
  "Cloud Run Ingress Settings" \
  "internal"

check_policy "run.allowedVPCEgress" \
  "Cloud Run VPC Egress Settings" \
  "private-ranges-only"

check_policy "cloudfunctions.requireVPCConnector" \
  "Require VPC Connector for Cloud Functions" \
  ""

# Check important policies
echo -e "${BLUE}=== Important Policies ===${NC}\n"

check_policy "iam.allowedPolicyMemberDomains" \
  "Allowed IAM Policy Member Domains" \
  ""

check_policy "compute.trustedImageProjects" \
  "Trusted Image Projects (for VPC Connector)" \
  ""

# Check moderate impact policies
echo -e "${BLUE}=== Moderate Impact Policies ===${NC}\n"

check_policy "compute.vmExternalIpAccess" \
  "VM External IP Access" \
  ""

check_policy "compute.requireShieldedVm" \
  "Require Shielded VMs" \
  ""

# Check for VPC Service Controls
echo -e "${BLUE}=== VPC Service Controls ===${NC}\n"
echo -e "${YELLOW}Checking: VPC Service Controls Perimeters${NC}"

if command -v gcloud &> /dev/null; then
  # Check if access-context-manager API is enabled
  ACM_ENABLED=$(gcloud services list --enabled --filter="name:accesscontextmanager.googleapis.com" --format="value(name)" 2>/dev/null || echo "")

  if [ -n "$ACM_ENABLED" ]; then
    # Get access policy
    POLICY_NAME=$(gcloud access-context-manager policies list --format="value(name)" 2>/dev/null | head -n1 || echo "")

    if [ -n "$POLICY_NAME" ]; then
      echo "  Access Context Manager Policy: $POLICY_NAME"

      # List perimeters
      PERIMETERS=$(gcloud access-context-manager perimeters list \
        --policy=$POLICY_NAME \
        --format="value(name)" 2>/dev/null || echo "")

      if [ -n "$PERIMETERS" ]; then
        echo "  Found perimeters:"
        echo "$PERIMETERS" | while read -r perimeter; do
          echo "    - $perimeter"
        done

        # Check if project is in any perimeter
        PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)" 2>/dev/null)
        IN_PERIMETER=false

        echo "$PERIMETERS" | while read -r perimeter; do
          RESOURCES=$(gcloud access-context-manager perimeters describe "$perimeter" \
            --policy=$POLICY_NAME \
            --format="value(status.resources[])" 2>/dev/null || echo "")

          if echo "$RESOURCES" | grep -q "$PROJECT_NUMBER"; then
            IN_PERIMETER=true
            echo -e "  ${MAGENTA}⚠ WARNING: Project is in VPC-SC perimeter '$perimeter'${NC}"
            echo "    Additional configuration required for Cloud Functions"
            WARNINGS=$((WARNINGS + 1))
          fi
        done

        if [ "$IN_PERIMETER" = "false" ]; then
          echo -e "  ${GREEN}✓ Project not in any VPC-SC perimeter${NC}"
        fi
      else
        echo -e "  ${GREEN}✓ No VPC Service Controls perimeters found${NC}"
      fi
    else
      echo -e "  ${GREEN}✓ No Access Context Manager policy configured${NC}"
    fi
  else
    echo -e "  ${GREEN}✓ Access Context Manager API not enabled${NC}"
  fi
else
  echo -e "  ${YELLOW}⚠ Unable to check (gcloud not available)${NC}"
fi

echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Compliance Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

COMPLIANCE_PERCENTAGE=$((COMPLIANCE_SCORE * 100 / TOTAL_CHECKS))

echo "Total Checks: $TOTAL_CHECKS"
echo "Compliant: $COMPLIANCE_SCORE"
echo "Warnings: $WARNINGS"
echo "Errors: $ERRORS"
echo ""

if [ $ERRORS -eq 0 ] && [ $WARNINGS -eq 0 ]; then
  echo -e "${GREEN}✓ FULLY COMPLIANT${NC}"
  echo "  All organization policies allow private Cloud Functions deployment"
  echo ""
  echo -e "${GREEN}Next Steps:${NC}"
  echo "  1. Proceed with VPC infrastructure setup: ./scripts/deployment/setup_vpc_infrastructure.sh"
  echo "  2. Deploy private Cloud Functions: ./scripts/deployment/deploy_private_cloud_functions.sh"
elif [ $ERRORS -eq 0 ]; then
  echo -e "${YELLOW}⚠ COMPLIANT WITH WARNINGS${NC}"
  echo "  Deployment should succeed, but review warnings above"
  echo ""
  echo -e "${YELLOW}Recommended Actions:${NC}"
  echo "  1. Review warnings and plan for additional configuration"
  echo "  2. If in VPC-SC perimeter, coordinate with security team"
  echo "  3. Test deployment in non-production environment first"
  echo "  4. Proceed with caution: ./scripts/deployment/deploy_private_cloud_functions.sh"
else
  echo -e "${RED}✗ NOT COMPLIANT${NC}"
  echo "  One or more policies will block deployment"
  echo ""
  echo -e "${RED}Required Actions:${NC}"
  echo "  1. Review errors above and identify blocking policies"
  echo "  2. Request policy exceptions from GCP organization admin"
  echo "  3. See: docs/investigations/ORGANIZATION_POLICY_COMPLIANCE.md"
  echo "  4. DO NOT attempt deployment until policies are resolved"
fi

echo ""

# Check specific VPC Connector requirement
echo -e "${BLUE}=== VPC Connector Recommendation ===${NC}\n"

VPC_CONNECTOR_REQUIRED=$(gcloud resource-manager org-policies describe \
  cloudfunctions.requireVPCConnector \
  --project=$PROJECT_ID \
  --effective \
  --format=json 2>/dev/null | jq -r '.booleanPolicy.enforced // false')

if [ "$VPC_CONNECTOR_REQUIRED" = "true" ]; then
  echo -e "${YELLOW}⚠ VPC Connector is REQUIRED by organization policy${NC}"
  echo ""
  echo "Deployment configuration:"
  echo "  1. Create VPC Connector: ./scripts/deployment/setup_vpc_infrastructure.sh"
  echo "  2. Use connector in deployment (not Direct VPC egress)"
  echo "  3. Expected additional cost: \$40-200/month"
else
  echo -e "${GREEN}✓ VPC Connector is OPTIONAL${NC}"
  echo ""
  echo "Recommended deployment configuration:"
  echo "  - Use Direct VPC egress (no additional cost)"
  echo "  - Better performance than VPC Connector"
  echo "  - Simpler architecture"
  echo ""
  echo "Alternative (if needed for cross-project/VPN):"
  echo "  - Create VPC Connector: ./scripts/deployment/setup_vpc_infrastructure.sh"
fi

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Organization policy check complete!${NC}"
echo -e "${BLUE}========================================${NC}"

# Exit with appropriate code
if [ $ERRORS -gt 0 ]; then
  exit 1
else
  exit 0
fi
