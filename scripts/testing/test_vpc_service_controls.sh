#!/bin/bash
# Test VPC Service Controls Configuration and Compliance
# This script validates VPC-SC perimeter, policies, and monitors for violations

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
PROJECT_NUMBER=$(gcloud projects describe $PROJECT_ID --format="value(projectNumber)" 2>/dev/null)
ORG_ID=$(gcloud projects describe $PROJECT_ID --format="value(parent.id)" 2>/dev/null || echo "")
PERIMETER_NAME=${PERIMETER_NAME:-"security-agent-perimeter"}
ACCESS_LEVEL_NAME=${ACCESS_LEVEL_NAME:-"security-agent-access"}

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VPC Service Controls Test Suite${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Project ID:${NC} $PROJECT_ID"
echo -e "${GREEN}Project Number:${NC} $PROJECT_NUMBER"
echo -e "${GREEN}Organization ID:${NC} ${ORG_ID:-"N/A"}"
echo ""

TESTS_PASSED=0
TESTS_FAILED=0
TESTS_WARNING=0

# Function to run a test
run_test() {
  local TEST_NAME=$1
  local TEST_COMMAND=$2
  local EXPECTED_RESULT=$3

  echo -e "${YELLOW}Test: ${TEST_NAME}${NC}"

  if eval "$TEST_COMMAND"; then
    if [ "$EXPECTED_RESULT" = "pass" ]; then
      echo -e "${GREEN}✓ PASS${NC}\n"
      TESTS_PASSED=$((TESTS_PASSED + 1))
      return 0
    else
      echo -e "${YELLOW}⚠ WARNING: Test passed but expected failure${NC}\n"
      TESTS_WARNING=$((TESTS_WARNING + 1))
      return 1
    fi
  else
    if [ "$EXPECTED_RESULT" = "fail" ]; then
      echo -e "${GREEN}✓ PASS (expected failure)${NC}\n"
      TESTS_PASSED=$((TESTS_PASSED + 1))
      return 0
    else
      echo -e "${RED}✗ FAIL${NC}\n"
      TESTS_FAILED=$((TESTS_FAILED + 1))
      return 1
    fi
  fi
}

# Check if organization exists
if [ -z "$ORG_ID" ] || [ "$ORG_ID" = "null" ]; then
  echo -e "${YELLOW}⚠ WARNING: Project is not part of an organization${NC}"
  echo "  VPC Service Controls requires an organization."
  echo "  Skipping VPC-SC tests."
  exit 0
fi

# Get policy name
POLICY_NAME=$(gcloud access-context-manager policies list \
  --organization=$ORG_ID \
  --format="value(name)" 2>/dev/null | head -n1 || echo "")

if [ -z "$POLICY_NAME" ]; then
  echo -e "${YELLOW}⚠ WARNING: No Access Context Manager policy found${NC}"
  echo "  VPC Service Controls not configured for this organization."
  echo "  Run: ./scripts/deployment/setup_vpc_service_controls.sh"
  exit 0
fi

echo -e "${GREEN}Policy Name:${NC} $POLICY_NAME\n"

# Test 1: Verify perimeter exists
echo -e "${BLUE}=== Test 1: Perimeter Configuration ===${NC}\n"

run_test "Perimeter '$PERIMETER_NAME' exists" \
  "gcloud access-context-manager perimeters describe $PERIMETER_NAME --policy=$POLICY_NAME &>/dev/null" \
  "pass"

if [ $? -eq 0 ]; then
  # Get perimeter details
  PERIMETER_STATUS=$(gcloud access-context-manager perimeters describe $PERIMETER_NAME \
    --policy=$POLICY_NAME \
    --format=json 2>/dev/null)

  # Check if in dry-run or enforced
  HAS_STATUS=$(echo "$PERIMETER_STATUS" | jq -r '.status // empty')
  HAS_SPEC=$(echo "$PERIMETER_STATUS" | jq -r '.spec // empty')

  if [ -n "$HAS_STATUS" ] && [ "$HAS_STATUS" != "null" ]; then
    echo -e "${GREEN}Perimeter Mode: ENFORCED${NC}"
    echo "  ⚠️ Perimeter is actively blocking non-compliant traffic"
  elif [ -n "$HAS_SPEC" ] && [ "$HAS_SPEC" != "null" ]; then
    echo -e "${YELLOW}Perimeter Mode: DRY-RUN${NC}"
    echo "  ℹ️ Perimeter is logging violations but not blocking"
  else
    echo -e "${MAGENTA}Perimeter Mode: UNKNOWN${NC}"
  fi
  echo ""

  # Check protected projects
  PROJECTS=$(echo "$PERIMETER_STATUS" | jq -r '.spec.resources[] // .status.resources[] // empty' 2>/dev/null)
  if echo "$PROJECTS" | grep -q "$PROJECT_NUMBER"; then
    echo -e "${GREEN}✓ Project is in perimeter${NC}"
  else
    echo -e "${RED}✗ Project NOT in perimeter${NC}"
    TESTS_FAILED=$((TESTS_FAILED + 1))
  fi
  echo ""

  # Check restricted services
  SERVICES=$(echo "$PERIMETER_STATUS" | jq -r '.spec.restrictedServices[] // .status.restrictedServices[] // empty' 2>/dev/null)
  echo "Protected Services:"
  echo "$SERVICES" | while read -r service; do
    echo "  - $service"
  done
  echo ""
fi

# Test 2: Verify access level exists
echo -e "${BLUE}=== Test 2: Access Level Configuration ===${NC}\n"

run_test "Access level '$ACCESS_LEVEL_NAME' exists" \
  "gcloud access-context-manager levels describe $ACCESS_LEVEL_NAME --policy=$POLICY_NAME &>/dev/null" \
  "pass"

if [ $? -eq 0 ]; then
  ACCESS_LEVEL=$(gcloud access-context-manager levels describe $ACCESS_LEVEL_NAME \
    --policy=$POLICY_NAME \
    --format=json 2>/dev/null)

  # Check for service accounts
  SCC_SA="service-${PROJECT_NUMBER}@gcp-sa-computescanning.iam.gserviceaccount.com"
  SCHEDULER_SA="${PROJECT_NUMBER}-compute@developer.gserviceaccount.com"
  FETCHER_SA="security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com"

  echo "Checking service accounts in access level..."

  if echo "$ACCESS_LEVEL" | jq -r '.basic.conditions[].members[]' | grep -q "$SCC_SA"; then
    echo -e "${GREEN}✓ Security Command Center P4SA included${NC}"
  else
    echo -e "${YELLOW}⚠ Security Command Center P4SA missing${NC}"
    TESTS_WARNING=$((TESTS_WARNING + 1))
  fi

  if echo "$ACCESS_LEVEL" | jq -r '.basic.conditions[].members[]' | grep -q "$SCHEDULER_SA"; then
    echo -e "${GREEN}✓ Cloud Scheduler SA included${NC}"
  else
    echo -e "${YELLOW}⚠ Cloud Scheduler SA missing${NC}"
    TESTS_WARNING=$((TESTS_WARNING + 1))
  fi

  if echo "$ACCESS_LEVEL" | jq -r '.basic.conditions[].members[]' | grep -q "$FETCHER_SA"; then
    echo -e "${GREEN}✓ Cloud Functions SA included${NC}"
  else
    echo -e "${YELLOW}⚠ Cloud Functions SA missing${NC}"
    TESTS_WARNING=$((TESTS_WARNING + 1))
  fi

  echo ""
fi

# Test 3: Check for VPC-SC violations
echo -e "${BLUE}=== Test 3: VPC Service Controls Violations ===${NC}\n"

echo "Checking for VPC-SC violations in last 24 hours..."

VIOLATIONS=$(gcloud logging read \
  'protoPayload.metadata.vpcServiceControlsUniqueId:* AND timestamp>="'$(date -u -d '24 hours ago' +%Y-%m-%dT%H:%M:%S)'"' \
  --limit=100 \
  --format=json 2>/dev/null | jq -r '.[] | select(.protoPayload.metadata.vpcServiceControlsUniqueId != null)')

VIOLATION_COUNT=$(echo "$VIOLATIONS" | jq -s 'length' 2>/dev/null || echo "0")

if [ "$VIOLATION_COUNT" -eq 0 ]; then
  echo -e "${GREEN}✓ No VPC-SC violations found${NC}"
  echo "  Perimeter configuration is working correctly"
  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${YELLOW}⚠ Found $VIOLATION_COUNT VPC-SC violations${NC}"
  echo ""
  echo "Recent violations:"
  echo "$VIOLATIONS" | jq -r 'limit(5; . | "\(.timestamp) - \(.protoPayload.metadata.violationReason // "Unknown reason") - \(.protoPayload.resourceName // "Unknown resource")")'
  echo ""
  echo "Full details:"
  echo "  gcloud logging read 'protoPayload.metadata.vpcServiceControlsUniqueId:*' --limit=50"
  TESTS_WARNING=$((TESTS_WARNING + 1))
fi

echo ""

# Test 4: Verify Cloud Functions deployment
echo -e "${BLUE}=== Test 4: Cloud Functions Integration ===${NC}\n"

FUNCTION_NAME="unified-security-fetcher"
REGION="us-central1"

if gcloud functions describe $FUNCTION_NAME --region=$REGION --project=$PROJECT_ID &>/dev/null; then
  echo -e "${GREEN}✓ Cloud Function exists${NC}"

  # Check if function can access BigQuery
  echo ""
  echo "Testing Cloud Function access to BigQuery..."

  # Trigger function via Cloud Scheduler
  JOB_NAME="unified-security-findings-trigger"

  if gcloud scheduler jobs describe $JOB_NAME --location=$REGION --project=$PROJECT_ID &>/dev/null; then
    echo "Triggering Cloud Scheduler job..."

    if gcloud scheduler jobs run $JOB_NAME --location=$REGION --project=$PROJECT_ID 2>/dev/null; then
      echo -e "${GREEN}✓ Cloud Scheduler triggered successfully${NC}"

      # Wait a bit for execution
      sleep 10

      # Check function logs for errors
      RECENT_ERRORS=$(gcloud functions logs read $FUNCTION_NAME \
        --region=$REGION \
        --limit=20 \
        --format=json 2>/dev/null | jq -r '.[] | select(.severity == "ERROR")')

      if [ -z "$RECENT_ERRORS" ]; then
        echo -e "${GREEN}✓ No errors in function logs${NC}"
        TESTS_PASSED=$((TESTS_PASSED + 1))
      else
        echo -e "${YELLOW}⚠ Errors found in function logs${NC}"
        echo "$RECENT_ERRORS" | jq -r 'limit(3; .textPayload // .jsonPayload)'
        TESTS_WARNING=$((TESTS_WARNING + 1))
      fi
    else
      echo -e "${YELLOW}⚠ Failed to trigger Cloud Scheduler job${NC}"
      TESTS_WARNING=$((TESTS_WARNING + 1))
    fi
  else
    echo -e "${YELLOW}⚠ Cloud Scheduler job not found${NC}"
    TESTS_WARNING=$((TESTS_WARNING + 1))
  fi
else
  echo -e "${YELLOW}⚠ Cloud Function not deployed${NC}"
  echo "  Run: ./scripts/deployment/deploy_private_cloud_functions.sh"
  TESTS_WARNING=$((TESTS_WARNING + 1))
fi

echo ""

# Test 5: Check ingress policies
echo -e "${BLUE}=== Test 5: Ingress Policies ===${NC}\n"

INGRESS_POLICIES=$(gcloud access-context-manager perimeters describe $PERIMETER_NAME \
  --policy=$POLICY_NAME \
  --format=json 2>/dev/null | jq -r '.spec.ingressPolicies // .status.ingressPolicies // []')

INGRESS_COUNT=$(echo "$INGRESS_POLICIES" | jq 'length' 2>/dev/null || echo "0")

if [ "$INGRESS_COUNT" -gt 0 ]; then
  echo -e "${GREEN}✓ Found $INGRESS_COUNT ingress policies${NC}"

  # Check for key ingress policies
  if echo "$INGRESS_POLICIES" | jq -r '.[].ingressTo.operations[].serviceName' | grep -q "run.googleapis.com"; then
    echo -e "${GREEN}✓ Cloud Run (Cloud Functions) ingress configured${NC}"
  else
    echo -e "${YELLOW}⚠ Cloud Run ingress policy missing${NC}"
  fi

  if echo "$INGRESS_POLICIES" | jq -r '.[].ingressTo.operations[].serviceName' | grep -q "bigquery.googleapis.com"; then
    echo -e "${GREEN}✓ BigQuery ingress configured${NC}"
  else
    echo -e "${YELLOW}⚠ BigQuery ingress policy missing${NC}"
  fi

  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${YELLOW}⚠ No ingress policies configured${NC}"
  TESTS_WARNING=$((TESTS_WARNING + 1))
fi

echo ""

# Test 6: Check egress policies
echo -e "${BLUE}=== Test 6: Egress Policies ===${NC}\n"

EGRESS_POLICIES=$(gcloud access-context-manager perimeters describe $PERIMETER_NAME \
  --policy=$POLICY_NAME \
  --format=json 2>/dev/null | jq -r '.spec.egressPolicies // .status.egressPolicies // []')

EGRESS_COUNT=$(echo "$EGRESS_POLICIES" | jq 'length' 2>/dev/null || echo "0")

if [ "$EGRESS_COUNT" -gt 0 ]; then
  echo -e "${GREEN}✓ Found $EGRESS_COUNT egress policies${NC}"

  # Check for key egress policies
  if echo "$EGRESS_POLICIES" | jq -r '.[].egressTo.operations[].serviceName' | grep -q "bigquery.googleapis.com"; then
    echo -e "${GREEN}✓ BigQuery egress configured${NC}"
  else
    echo -e "${YELLOW}⚠ BigQuery egress policy missing${NC}"
  fi

  if echo "$EGRESS_POLICIES" | jq -r '.[].egressTo.operations[].serviceName' | grep -q "securitycenter.googleapis.com"; then
    echo -e "${GREEN}✓ Security Command Center egress configured${NC}"
  else
    echo -e "${YELLOW}⚠ Security Command Center egress policy missing${NC}"
  fi

  TESTS_PASSED=$((TESTS_PASSED + 1))
else
  echo -e "${YELLOW}⚠ No egress policies configured${NC}"
  TESTS_WARNING=$((TESTS_WARNING + 1))
fi

echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Test Summary${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

TOTAL_TESTS=$((TESTS_PASSED + TESTS_FAILED + TESTS_WARNING))

echo "Total Tests: $TOTAL_TESTS"
echo -e "${GREEN}Passed: $TESTS_PASSED${NC}"
echo -e "${YELLOW}Warnings: $TESTS_WARNING${NC}"
echo -e "${RED}Failed: $TESTS_FAILED${NC}"
echo ""

if [ $TESTS_FAILED -eq 0 ] && [ $TESTS_WARNING -eq 0 ]; then
  echo -e "${GREEN}✓ ALL TESTS PASSED${NC}"
  echo "  VPC Service Controls is properly configured"
  echo ""
  echo -e "${GREEN}Next Steps:${NC}"
  echo "  1. Monitor for violations: gcloud logging read 'protoPayload.metadata.vpcServiceControlsUniqueId:*' --limit=50"
  echo "  2. If in dry-run mode and no violations, consider enforcing:"
  echo "     gcloud access-context-manager perimeters dry-run enforce $PERIMETER_NAME --policy=$POLICY_NAME"
elif [ $TESTS_FAILED -eq 0 ]; then
  echo -e "${YELLOW}⚠ TESTS PASSED WITH WARNINGS${NC}"
  echo "  Review warnings above and address as needed"
  echo ""
  echo -e "${YELLOW}Recommended Actions:${NC}"
  echo "  1. Review warnings and determine if action needed"
  echo "  2. Check VPC-SC violation logs for details"
  echo "  3. Update ingress/egress policies if needed"
  echo "  4. Continue monitoring in dry-run mode"
else
  echo -e "${RED}✗ TESTS FAILED${NC}"
  echo "  VPC Service Controls configuration has issues"
  echo ""
  echo -e "${RED}Required Actions:${NC}"
  echo "  1. Review failed tests above"
  echo "  2. Check VPC-SC configuration"
  echo "  3. Fix issues before enforcing perimeter"
  echo "  4. Re-run tests: ./scripts/testing/test_vpc_service_controls.sh"
fi

echo ""

# Exit with appropriate code
if [ $TESTS_FAILED -gt 0 ]; then
  exit 1
else
  exit 0
fi
