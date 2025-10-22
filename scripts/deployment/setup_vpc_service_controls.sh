#!/bin/bash
# Setup VPC Service Controls for Private Cloud Functions
# This script creates VPC-SC perimeters and policies for secure Cloud Functions deployment

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
echo -e "${BLUE}VPC Service Controls Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Project ID:${NC} $PROJECT_ID"
echo -e "${GREEN}Project Number:${NC} $PROJECT_NUMBER"
echo -e "${GREEN}Organization ID:${NC} ${ORG_ID:-"N/A"}"
echo ""

# Check if organization exists
if [ -z "$ORG_ID" ] || [ "$ORG_ID" = "null" ]; then
  echo -e "${RED}✗ ERROR: Project is not part of an organization${NC}"
  echo "  VPC Service Controls requires an organization."
  echo "  This project cannot use VPC Service Controls."
  exit 1
fi

# Step 1: Enable required APIs
echo -e "${YELLOW}Step 1: Enabling VPC Service Controls APIs...${NC}"
gcloud services enable \
  accesscontextmanager.googleapis.com \
  cloudfunctions.googleapis.com \
  run.googleapis.com \
  cloudscheduler.googleapis.com \
  bigquery.googleapis.com \
  securitycenter.googleapis.com \
  compute.googleapis.com \
  storage.googleapis.com \
  iam.googleapis.com \
  --project=$PROJECT_ID

echo -e "${GREEN}✓ APIs enabled${NC}\n"

# Step 2: Check for existing Access Context Manager policy
echo -e "${YELLOW}Step 2: Checking Access Context Manager policy...${NC}"

POLICY_NAME=$(gcloud access-context-manager policies list \
  --organization=$ORG_ID \
  --format="value(name)" 2>/dev/null | head -n1 || echo "")

if [ -z "$POLICY_NAME" ]; then
  echo "No Access Context Manager policy found. Creating one..."
  echo ""
  echo -e "${MAGENTA}NOTE: This requires Organization Administrator role.${NC}"
  echo "If this fails, ask your org admin to create an access policy."
  echo ""
  read -p "Attempt to create Access Context Manager policy? (y/N): " CREATE_POLICY

  if [[ "$CREATE_POLICY" =~ ^[Yy]$ ]]; then
    POLICY_TITLE="VPC Service Controls Policy - $(date +%Y%m%d)"

    gcloud access-context-manager policies create \
      --title="$POLICY_TITLE" \
      --organization=$ORG_ID

    POLICY_NAME=$(gcloud access-context-manager policies list \
      --organization=$ORG_ID \
      --format="value(name)" | head -n1)

    echo -e "${GREEN}✓ Access Context Manager policy created${NC}"
  else
    echo -e "${RED}✗ Cannot proceed without Access Context Manager policy${NC}"
    echo "  Contact your organization administrator to create a policy."
    exit 1
  fi
else
  echo -e "${GREEN}✓ Access Context Manager policy exists${NC}"
  echo "  Policy: $POLICY_NAME"
fi

echo ""

# Step 3: Create or update access level
echo -e "${YELLOW}Step 3: Creating access level for Security Command Center...${NC}"

# Define access level for Security Command Center P4SA
cat > /tmp/access-level-${ACCESS_LEVEL_NAME}.yaml <<EOF
name: ${POLICY_NAME}/accessLevels/${ACCESS_LEVEL_NAME}
title: "Security Agent Access Level"
description: "Access level for Security Agent Cloud Functions and SCC scanning"
basic:
  conditions:
    - members:
        - serviceAccount:service-${PROJECT_NUMBER}@gcp-sa-computescanning.iam.gserviceaccount.com
        - serviceAccount:${PROJECT_NUMBER}-compute@developer.gserviceaccount.com
        - serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com
EOF

# Check if access level exists
if gcloud access-context-manager levels describe "$ACCESS_LEVEL_NAME" \
  --policy=$POLICY_NAME &>/dev/null; then
  echo "Updating existing access level..."
  gcloud access-context-manager levels update "$ACCESS_LEVEL_NAME" \
    --policy=$POLICY_NAME \
    --basic-level-spec=/tmp/access-level-${ACCESS_LEVEL_NAME}.yaml
else
  echo "Creating new access level..."
  gcloud access-context-manager levels create "$ACCESS_LEVEL_NAME" \
    --policy=$POLICY_NAME \
    --basic-level-spec=/tmp/access-level-${ACCESS_LEVEL_NAME}.yaml
fi

rm -f /tmp/access-level-${ACCESS_LEVEL_NAME}.yaml

echo -e "${GREEN}✓ Access level created/updated${NC}\n"

# Step 4: Create service perimeter (dry-run mode first)
echo -e "${YELLOW}Step 4: Creating VPC Service Controls perimeter...${NC}"
echo ""
echo "This will create a service perimeter in DRY-RUN mode first."
echo "Dry-run mode allows testing without blocking traffic."
echo ""

# Services to protect
RESTRICTED_SERVICES=(
  "bigquery.googleapis.com"
  "cloudscheduler.googleapis.com"
  "run.googleapis.com"
  "securitycenter.googleapis.com"
  "storage.googleapis.com"
  "compute.googleapis.com"
)

# Convert array to comma-separated string
SERVICES_STRING=$(IFS=,; echo "${RESTRICTED_SERVICES[*]}")

# Check if perimeter exists
if gcloud access-context-manager perimeters describe "$PERIMETER_NAME" \
  --policy=$POLICY_NAME &>/dev/null; then
  echo -e "${YELLOW}⚠ Perimeter '$PERIMETER_NAME' already exists${NC}"
  echo ""
  read -p "Update existing perimeter? (y/N): " UPDATE_PERIMETER

  if [[ "$UPDATE_PERIMETER" =~ ^[Yy]$ ]]; then
    echo "Updating perimeter in dry-run mode..."

    gcloud access-context-manager perimeters dry-run update "$PERIMETER_NAME" \
      --policy=$POLICY_NAME \
      --add-resources=projects/$PROJECT_NUMBER \
      --add-restricted-services=$SERVICES_STRING \
      --add-access-levels=$ACCESS_LEVEL_NAME \
      --enable-vpc-accessible-services \
      --vpc-allowed-services=$SERVICES_STRING

    echo -e "${GREEN}✓ Perimeter updated (dry-run mode)${NC}"
  else
    echo "Skipping perimeter update."
  fi
else
  echo "Creating new perimeter in dry-run mode..."

  gcloud access-context-manager perimeters dry-run create "$PERIMETER_NAME" \
    --title="Security Agent Service Perimeter" \
    --policy=$POLICY_NAME \
    --resources=projects/$PROJECT_NUMBER \
    --restricted-services=$SERVICES_STRING \
    --access-levels=$ACCESS_LEVEL_NAME \
    --enable-vpc-accessible-services \
    --vpc-allowed-services=$SERVICES_STRING \
    --perimeter-type=regular

  echo -e "${GREEN}✓ Perimeter created (dry-run mode)${NC}"
fi

echo ""

# Step 5: Create ingress policies
echo -e "${YELLOW}Step 5: Creating ingress policies...${NC}"

# Ingress policy for Cloud Scheduler to Cloud Functions
cat > /tmp/ingress-policy-scheduler.yaml <<EOF
- ingressFrom:
    identityType: ANY_SERVICE_ACCOUNT
    sources:
      - resource: projects/$PROJECT_NUMBER
  ingressTo:
    operations:
      - serviceName: run.googleapis.com
        methodSelectors:
          - method: "*"
    resources:
      - projects/$PROJECT_NUMBER
EOF

# Ingress policy for BigQuery access
cat > /tmp/ingress-policy-bigquery.yaml <<EOF
- ingressFrom:
    identityType: ANY_SERVICE_ACCOUNT
    sources:
      - accessLevel: ${POLICY_NAME}/accessLevels/${ACCESS_LEVEL_NAME}
  ingressTo:
    operations:
      - serviceName: bigquery.googleapis.com
        methodSelectors:
          - method: "*"
    resources:
      - projects/$PROJECT_NUMBER
EOF

# Ingress policy for Security Command Center
cat > /tmp/ingress-policy-scc.yaml <<EOF
- ingressFrom:
    identityType: ANY_SERVICE_ACCOUNT
    sources:
      - accessLevel: ${POLICY_NAME}/accessLevels/${ACCESS_LEVEL_NAME}
  ingressTo:
    operations:
      - serviceName: securitycenter.googleapis.com
        methodSelectors:
          - method: "*"
    resources:
      - projects/$PROJECT_NUMBER
EOF

echo "Updating perimeter with ingress policies..."

# Combine ingress policies
cat /tmp/ingress-policy-*.yaml > /tmp/combined-ingress-policies.yaml

# Update perimeter with ingress rules
gcloud access-context-manager perimeters dry-run update "$PERIMETER_NAME" \
  --policy=$POLICY_NAME \
  --set-ingress-policies=/tmp/combined-ingress-policies.yaml

rm -f /tmp/ingress-policy-*.yaml /tmp/combined-ingress-policies.yaml

echo -e "${GREEN}✓ Ingress policies configured${NC}\n"

# Step 6: Create egress policies
echo -e "${YELLOW}Step 6: Creating egress policies...${NC}"

# Egress policy for Cloud Functions to access BigQuery
cat > /tmp/egress-policy-bigquery.yaml <<EOF
- egressFrom:
    identities:
      - serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com
  egressTo:
    operations:
      - serviceName: bigquery.googleapis.com
        methodSelectors:
          - method: "*"
    resources:
      - projects/$PROJECT_NUMBER
EOF

# Egress policy for Cloud Functions to access Security Command Center
cat > /tmp/egress-policy-scc.yaml <<EOF
- egressFrom:
    identities:
      - serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com
  egressTo:
    operations:
      - serviceName: securitycenter.googleapis.com
        methodSelectors:
          - method: "*"
    resources:
      - projects/$PROJECT_NUMBER
EOF

# Egress policy for Cloud Functions to access Compute/Storage
cat > /tmp/egress-policy-gcp-apis.yaml <<EOF
- egressFrom:
    identities:
      - serviceAccount:security-fetcher-sa@${PROJECT_ID}.iam.gserviceaccount.com
  egressTo:
    operations:
      - serviceName: compute.googleapis.com
      - serviceName: storage.googleapis.com
      - serviceName: iam.googleapis.com
    resources:
      - projects/$PROJECT_NUMBER
EOF

echo "Updating perimeter with egress policies..."

# Combine egress policies
cat /tmp/egress-policy-*.yaml > /tmp/combined-egress-policies.yaml

# Update perimeter with egress rules
gcloud access-context-manager perimeters dry-run update "$PERIMETER_NAME" \
  --policy=$POLICY_NAME \
  --set-egress-policies=/tmp/combined-egress-policies.yaml

rm -f /tmp/egress-policy-*.yaml /tmp/combined-egress-policies.yaml

echo -e "${GREEN}✓ Egress policies configured${NC}\n"

# Step 7: Test and enforce
echo -e "${YELLOW}Step 7: Testing and enforcement...${NC}"
echo ""
echo -e "${MAGENTA}IMPORTANT: Perimeter is currently in DRY-RUN mode${NC}"
echo ""
echo "DRY-RUN mode means:"
echo "  - VPC-SC logs access violations but doesn't block them"
echo "  - You can test your Cloud Functions deployment safely"
echo "  - Monitor Cloud Logging for 'vpcServiceControlsUniqueId' to see violations"
echo ""
echo "Testing steps:"
echo "  1. Deploy your Cloud Functions: ./scripts/deployment/deploy_private_cloud_functions.sh"
echo "  2. Trigger Cloud Scheduler jobs"
echo "  3. Monitor logs for VPC-SC violations:"
echo "     gcloud logging read 'protoPayload.metadata.vpcServiceControlsUniqueId:*' --limit=50"
echo "  4. Fix any violations by updating ingress/egress policies"
echo "  5. Test for 24-48 hours"
echo ""
echo -e "${YELLOW}After successful testing, enforce the perimeter:${NC}"
echo ""
echo "  gcloud access-context-manager perimeters dry-run enforce $PERIMETER_NAME \\"
echo "    --policy=$POLICY_NAME"
echo ""

read -p "Do you want to ENFORCE the perimeter now? (y/N): " ENFORCE_NOW

if [[ "$ENFORCE_NOW" =~ ^[Yy]$ ]]; then
  echo ""
  echo -e "${RED}WARNING: Enforcing perimeter will BLOCK non-compliant traffic!${NC}"
  echo "Make sure you've tested thoroughly in dry-run mode."
  echo ""
  read -p "Are you absolutely sure? (yes/NO): " CONFIRM_ENFORCE

  if [ "$CONFIRM_ENFORCE" = "yes" ]; then
    echo "Enforcing perimeter..."

    gcloud access-context-manager perimeters dry-run enforce "$PERIMETER_NAME" \
      --policy=$POLICY_NAME

    echo -e "${GREEN}✓ Perimeter ENFORCED${NC}"
    echo ""
    echo -e "${MAGENTA}Monitor closely for the next few hours!${NC}"
    echo "  Check logs: gcloud logging read 'severity>=ERROR' --limit=100"
  else
    echo "Enforcement cancelled. Remaining in dry-run mode."
  fi
else
  echo "Perimeter remains in dry-run mode (recommended for initial setup)."
fi

echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VPC Service Controls Setup Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Created Resources:${NC}"
echo "  - Access Context Manager Policy: $POLICY_NAME"
echo "  - Access Level: $ACCESS_LEVEL_NAME"
echo "  - Service Perimeter: $PERIMETER_NAME (DRY-RUN mode)"
echo ""
echo -e "${GREEN}Protected Services:${NC}"
for service in "${RESTRICTED_SERVICES[@]}"; do
  echo "  - $service"
done
echo ""
echo -e "${GREEN}Ingress Policies:${NC}"
echo "  - Cloud Scheduler → Cloud Functions (run.googleapis.com)"
echo "  - Service Accounts → BigQuery (bigquery.googleapis.com)"
echo "  - SCC P4SA → Security Command Center (securitycenter.googleapis.com)"
echo ""
echo -e "${GREEN}Egress Policies:${NC}"
echo "  - Cloud Functions → BigQuery"
echo "  - Cloud Functions → Security Command Center"
echo "  - Cloud Functions → Compute/Storage/IAM APIs"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Deploy Cloud Functions: ./scripts/deployment/deploy_private_cloud_functions.sh"
echo "  2. Test all functionality thoroughly"
echo "  3. Monitor VPC-SC logs for violations:"
echo "     gcloud logging read 'protoPayload.metadata.vpcServiceControlsUniqueId:*' --limit=50"
echo "  4. After 24-48 hours of successful testing, enforce perimeter:"
echo "     gcloud access-context-manager perimeters dry-run enforce $PERIMETER_NAME --policy=$POLICY_NAME"
echo ""
echo -e "${GREEN}VPC Service Controls setup complete!${NC}"
