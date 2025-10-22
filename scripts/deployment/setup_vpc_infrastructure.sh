#!/bin/bash
# Setup VPC Infrastructure for Private Cloud Functions
# This script creates necessary VPC resources for internal-only cloud functions

set -e

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
PROJECT_ID=${GOOGLE_CLOUD_PROJECT:-$(gcloud config get-value project)}
REGION=${REGION:-us-central1}
VPC_NETWORK=${VPC_NETWORK:-default}
VPC_SUBNET=${VPC_SUBNET:-default}
CONNECTOR_NAME="security-agent-connector"
CONNECTOR_CIDR="10.8.0.0/28"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VPC Infrastructure Setup${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Project ID:${NC} $PROJECT_ID"
echo -e "${GREEN}Region:${NC} $REGION"
echo -e "${GREEN}VPC Network:${NC} $VPC_NETWORK"
echo ""

# Step 1: Enable required APIs
echo -e "${YELLOW}Step 1: Enabling VPC and networking APIs...${NC}"
gcloud services enable \
  compute.googleapis.com \
  vpcaccess.googleapis.com \
  servicenetworking.googleapis.com \
  --project=$PROJECT_ID

echo -e "${GREEN}✓ APIs enabled${NC}\n"

# Step 2: Create or verify VPC network
echo -e "${YELLOW}Step 2: Setting up VPC network...${NC}"

if gcloud compute networks describe $VPC_NETWORK --project=$PROJECT_ID &>/dev/null; then
  echo -e "${GREEN}✓ VPC network '$VPC_NETWORK' already exists${NC}"
else
  echo "Creating VPC network '$VPC_NETWORK'..."
  gcloud compute networks create $VPC_NETWORK \
    --subnet-mode=auto \
    --project=$PROJECT_ID
  echo -e "${GREEN}✓ VPC network created${NC}"
fi

echo ""

# Step 3: Verify subnet exists
echo -e "${YELLOW}Step 3: Verifying subnet configuration...${NC}"

if gcloud compute networks subnets describe $VPC_SUBNET --region=$REGION --project=$PROJECT_ID &>/dev/null; then
  SUBNET_CIDR=$(gcloud compute networks subnets describe $VPC_SUBNET --region=$REGION --project=$PROJECT_ID --format="value(ipCidrRange)")
  echo -e "${GREEN}✓ Subnet '$VPC_SUBNET' exists in $REGION${NC}"
  echo "  CIDR range: $SUBNET_CIDR"
else
  echo -e "${RED}✗ Subnet '$VPC_SUBNET' not found in $REGION${NC}"
  echo "  Creating subnet..."
  gcloud compute networks subnets create $VPC_SUBNET \
    --network=$VPC_NETWORK \
    --region=$REGION \
    --range=10.128.0.0/20 \
    --project=$PROJECT_ID
  echo -e "${GREEN}✓ Subnet created${NC}"
fi

echo ""

# Step 4: Create firewall rules
echo -e "${YELLOW}Step 4: Creating firewall rules...${NC}"

# Allow internal traffic from Cloud Functions
FIREWALL_RULE_NAME="allow-cloud-functions-internal"

if gcloud compute firewall-rules describe $FIREWALL_RULE_NAME --project=$PROJECT_ID &>/dev/null; then
  echo -e "${GREEN}✓ Firewall rule '$FIREWALL_RULE_NAME' already exists${NC}"
else
  echo "Creating firewall rule to allow internal traffic..."
  gcloud compute firewall-rules create $FIREWALL_RULE_NAME \
    --network=$VPC_NETWORK \
    --allow=tcp,udp,icmp \
    --source-ranges=10.128.0.0/20,10.8.0.0/28 \
    --description="Allow traffic from Cloud Functions via VPC" \
    --project=$PROJECT_ID
  echo -e "${GREEN}✓ Firewall rule created${NC}"
fi

# Allow IAP traffic (for debugging)
IAP_FIREWALL_RULE="allow-iap-traffic"

if gcloud compute firewall-rules describe $IAP_FIREWALL_RULE --project=$PROJECT_ID &>/dev/null; then
  echo -e "${GREEN}✓ IAP firewall rule already exists${NC}"
else
  echo "Creating firewall rule for Identity-Aware Proxy..."
  gcloud compute firewall-rules create $IAP_FIREWALL_RULE \
    --network=$VPC_NETWORK \
    --allow=tcp:22,tcp:3389 \
    --source-ranges=35.235.240.0/20 \
    --description="Allow IAP traffic for debugging" \
    --project=$PROJECT_ID
  echo -e "${GREEN}✓ IAP firewall rule created${NC}"
fi

echo ""

# Step 5: Optional - Create Serverless VPC Access Connector
echo -e "${YELLOW}Step 5: VPC Access Connector (Optional)...${NC}"
echo ""
echo "Note: Direct VPC egress is recommended for better performance and lower cost."
echo "VPC Connector is only needed for specific use cases (cross-project, VPN, etc.)"
echo ""

read -p "Do you want to create a VPC Access Connector? (y/N): " CREATE_CONNECTOR

if [[ "$CREATE_CONNECTOR" =~ ^[Yy]$ ]]; then
  if gcloud compute networks vpc-access connectors describe $CONNECTOR_NAME --region=$REGION --project=$PROJECT_ID &>/dev/null; then
    echo -e "${GREEN}✓ VPC Connector '$CONNECTOR_NAME' already exists${NC}"
  else
    echo "Creating VPC Access Connector (this may take 5-10 minutes)..."
    echo "  Name: $CONNECTOR_NAME"
    echo "  CIDR: $CONNECTOR_CIDR"
    echo "  Min instances: 2"
    echo "  Max instances: 10"
    echo ""

    gcloud compute networks vpc-access connectors create $CONNECTOR_NAME \
      --region=$REGION \
      --network=$VPC_NETWORK \
      --range=$CONNECTOR_CIDR \
      --min-instances=2 \
      --max-instances=10 \
      --project=$PROJECT_ID

    echo -e "${GREEN}✓ VPC Connector created${NC}"
    echo ""
    echo -e "${YELLOW}Cost Note:${NC} VPC Connector runs 2-10 instances continuously"
    echo "  Estimated cost: $40-200/month depending on usage"
  fi
else
  echo "Skipping VPC Connector creation (using Direct VPC egress)"
  echo ""
  echo -e "${GREEN}Recommended configuration:${NC}"
  echo "  --vpc-egress=private-ranges-only"
  echo "  --network=projects/${PROJECT_ID}/global/networks/${VPC_NETWORK}"
  echo "  --subnet=projects/${PROJECT_ID}/regions/${REGION}/subnetworks/${VPC_SUBNET}"
fi

echo ""

# Step 6: Create Cloud NAT (optional, for internet access)
echo -e "${YELLOW}Step 6: Cloud NAT (Optional)...${NC}"
echo ""
echo "Cloud NAT allows Cloud Functions to access internet while using VPC egress."
echo "Required if functions need to call external APIs (GitHub, third-party services, etc.)"
echo ""

read -p "Do you want to create Cloud NAT? (y/N): " CREATE_NAT

if [[ "$CREATE_NAT" =~ ^[Yy]$ ]]; then
  ROUTER_NAME="security-agent-router"
  NAT_NAME="security-agent-nat"

  # Create Cloud Router
  if gcloud compute routers describe $ROUTER_NAME --region=$REGION --project=$PROJECT_ID &>/dev/null; then
    echo -e "${GREEN}✓ Cloud Router already exists${NC}"
  else
    echo "Creating Cloud Router..."
    gcloud compute routers create $ROUTER_NAME \
      --network=$VPC_NETWORK \
      --region=$REGION \
      --project=$PROJECT_ID
    echo -e "${GREEN}✓ Cloud Router created${NC}"
  fi

  # Create Cloud NAT
  if gcloud compute routers nats describe $NAT_NAME --router=$ROUTER_NAME --region=$REGION --project=$PROJECT_ID &>/dev/null; then
    echo -e "${GREEN}✓ Cloud NAT already exists${NC}"
  else
    echo "Creating Cloud NAT..."
    gcloud compute routers nats create $NAT_NAME \
      --router=$ROUTER_NAME \
      --region=$REGION \
      --auto-allocate-nat-external-ips \
      --nat-all-subnet-ip-ranges \
      --project=$PROJECT_ID
    echo -e "${GREEN}✓ Cloud NAT created${NC}"
  fi
else
  echo "Skipping Cloud NAT creation"
  echo "  Functions will only access Google APIs and internal resources"
fi

echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}VPC Infrastructure Setup Complete${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${GREEN}Created Resources:${NC}"
echo "  - VPC Network: $VPC_NETWORK"
echo "  - Subnet: $VPC_SUBNET (region: $REGION)"
echo "  - Firewall Rules: Internal traffic + IAP access"

if [[ "$CREATE_CONNECTOR" =~ ^[Yy]$ ]]; then
  echo "  - VPC Connector: $CONNECTOR_NAME (CIDR: $CONNECTOR_CIDR)"
fi

if [[ "$CREATE_NAT" =~ ^[Yy]$ ]]; then
  echo "  - Cloud NAT: $NAT_NAME (via router: $ROUTER_NAME)"
fi

echo ""
echo -e "${GREEN}Recommended Cloud Function Configuration:${NC}"
echo ""

if [[ "$CREATE_CONNECTOR" =~ ^[Yy]$ ]]; then
  echo "  Using VPC Connector:"
  echo "    --vpc-connector=$CONNECTOR_NAME"
  echo "    --vpc-egress=private-ranges-only"
else
  echo "  Using Direct VPC Egress (Recommended):"
  echo "    --vpc-egress=private-ranges-only"
  echo "    --network=projects/${PROJECT_ID}/global/networks/${VPC_NETWORK}"
  echo "    --subnet=projects/${PROJECT_ID}/regions/${REGION}/subnetworks/${VPC_SUBNET}"
fi

echo "    --ingress-settings=internal-only"
echo "    --no-allow-unauthenticated"
echo ""
echo -e "${YELLOW}Next Steps:${NC}"
echo "  1. Deploy private cloud functions: ./scripts/deployment/deploy_private_cloud_functions.sh"
echo "  2. Test connectivity: ./scripts/testing/test_private_functions.sh"
echo "  3. Monitor network metrics in Cloud Console"
echo ""
echo -e "${GREEN}Setup complete!${NC}"
