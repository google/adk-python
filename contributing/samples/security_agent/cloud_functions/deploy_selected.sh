#!/bin/bash

################################################################################
#                                                                              #
#   Modular Cloud Functions Deployment                                        #
#   Deploy only the functions you need!                                       #
#                                                                              #
################################################################################

set -e

PROJECT_ID=${1:-${GOOGLE_CLOUD_PROJECT}}
REGION=${2:-us-central1}

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m'

print_header() {
    echo -e "${BLUE}"
    echo "╔════════════════════════════════════════════════════════════════╗"
    echo "║                                                                ║"
    echo "║        Cloud Functions - Modular Deployment                   ║"
    echo "║        Deploy only what you need!                             ║"
    echo "║                                                                ║"
    echo "╚════════════════════════════════════════════════════════════════╝"
    echo -e "${NC}"
}

print_header

if [ -z "$PROJECT_ID" ]; then
    echo -e "${RED}Error: Project ID not provided${NC}"
    echo "Usage: $0 <project-id> [region]"
    exit 1
fi

echo -e "${CYAN}Available Function Categories:${NC}"
echo ""
echo "  1. 🔒 IAM & Security (7 functions)"
echo "     • Custom roles, standard roles, IAM bindings"
echo "     • Service accounts, users, security findings"
echo "     • Firewall rules"
echo ""
echo "  2. ☁️  Infrastructure (2 functions)"
echo "     • Compute instances, storage buckets"
echo ""
echo "  3. 📰 Feeds & Documentation (3 functions)"
echo "     • GCP release notes, security feeds, Confluence"
echo ""
echo "  4. 🎯 Analysis (1 function)"
echo "     • MSA Analyzer (recommended!)"
echo ""
echo "  5. 🎁 Everything (all 13 functions)"
echo ""
echo "  6. 🎨 Custom selection"
echo ""
echo "  0. ❌ Exit"
echo ""

read -p "Select category (1-6, 0 to exit): " category

case $category in
    1)
        echo -e "${GREEN}Deploying IAM & Security functions...${NC}"
        FUNCTIONS=(
            "fetch_custom_roles"
            "fetch_standard_roles"
            "fetch_iam_accounts"
            "fetch_service_account_roles"
            "fetch_user_roles"
            "fetch_security_findings"
            "fetch_firewall_rules"
        )
        ;;
    2)
        echo -e "${GREEN}Deploying Infrastructure functions...${NC}"
        FUNCTIONS=(
            "fetch_compute_instances"
            "fetch_storage_buckets"
        )
        ;;
    3)
        echo -e "${GREEN}Deploying Feeds & Documentation functions...${NC}"
        FUNCTIONS=(
            "fetch_gcp_release_notes"
            "fetch_security_feeds"
            "confluence_sync"
        )
        ;;
    4)
        echo -e "${GREEN}Deploying MSA Analyzer...${NC}"
        cd msa_analyzer
        ./deploy_complete.sh "$PROJECT_ID" "$REGION"
        echo -e "${GREEN}✅ MSA Analyzer deployed!${NC}"
        exit 0
        ;;
    5)
        echo -e "${GREEN}Deploying ALL functions...${NC}"
        FUNCTIONS=(
            "fetch_compute_instances"
            "fetch_custom_roles"
            "fetch_firewall_rules"
            "fetch_gcp_release_notes"
            "fetch_iam_accounts"
            "fetch_security_feeds"
            "fetch_security_findings"
            "fetch_service_account_roles"
            "fetch_standard_roles"
            "fetch_storage_buckets"
            "fetch_user_roles"
            "confluence_sync"
        )
        # Deploy MSA separately (uses different script)
        echo -e "${YELLOW}Also deploying MSA Analyzer...${NC}"
        ;;
    6)
        echo -e "${CYAN}Available functions:${NC}"
        echo ""
        ALL_FUNCTIONS=(
            "fetch_compute_instances"
            "fetch_custom_roles"
            "fetch_firewall_rules"
            "fetch_gcp_release_notes"
            "fetch_iam_accounts"
            "fetch_security_feeds"
            "fetch_security_findings"
            "fetch_service_account_roles"
            "fetch_standard_roles"
            "fetch_storage_buckets"
            "fetch_user_roles"
            "confluence_sync"
            "msa_analyzer"
        )

        for i in "${!ALL_FUNCTIONS[@]}"; do
            echo "  $((i+1)). ${ALL_FUNCTIONS[$i]}"
        done
        echo ""
        echo "Enter function numbers separated by spaces (e.g., 1 3 5)"
        read -p "Selection: " -a selections

        FUNCTIONS=()
        for num in "${selections[@]}"; do
            idx=$((num-1))
            if [ $idx -ge 0 ] && [ $idx -lt ${#ALL_FUNCTIONS[@]} ]; then
                FUNCTIONS+=("${ALL_FUNCTIONS[$idx]}")
            fi
        done

        if [ ${#FUNCTIONS[@]} -eq 0 ]; then
            echo -e "${RED}No valid functions selected${NC}"
            exit 1
        fi
        ;;
    0)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo -e "${RED}Invalid selection${NC}"
        exit 1
        ;;
esac

echo ""
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo -e "${CYAN}  Deploying ${#FUNCTIONS[@]} function(s)${NC}"
echo -e "${CYAN}═══════════════════════════════════════════════════════════════${NC}"
echo ""

DEPLOYED=0
FAILED=0

for func in "${FUNCTIONS[@]}"; do
    echo -e "${YELLOW}▶ Deploying ${func}...${NC}"

    if [ ! -d "$func" ]; then
        echo -e "${RED}  ✗ Directory not found: $func${NC}"
        ((FAILED++))
        continue
    fi

    cd "$func"

    if [ -f "deploy.sh" ]; then
        if ./deploy.sh "$PROJECT_ID" "$REGION" 2>&1 | grep -q "Deploy complete\|successfully"; then
            echo -e "${GREEN}  ✓ ${func} deployed${NC}"
            ((DEPLOYED++))
        else
            echo -e "${RED}  ✗ ${func} deployment failed${NC}"
            ((FAILED++))
        fi
    else
        echo -e "${RED}  ✗ deploy.sh not found in $func${NC}"
        ((FAILED++))
    fi

    cd ..
    echo ""
done

# Deploy MSA if it was selected (category 5 or custom)
if [ "$category" = "5" ] || [[ " ${FUNCTIONS[@]} " =~ " msa_analyzer " ]]; then
    echo -e "${YELLOW}▶ Deploying msa_analyzer...${NC}"
    cd msa_analyzer
    if ./deploy_complete.sh "$PROJECT_ID" "$REGION" 2>&1 | grep -q "Setup complete"; then
        echo -e "${GREEN}  ✓ msa_analyzer deployed${NC}"
        ((DEPLOYED++))
    else
        echo -e "${RED}  ✗ msa_analyzer deployment failed${NC}"
        ((FAILED++))
    fi
    cd ..
fi

echo ""
echo -e "${BLUE}╔════════════════════════════════════════════════════════════════╗${NC}"
echo -e "${BLUE}║                    Deployment Summary                         ║${NC}"
echo -e "${BLUE}╚════════════════════════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${GREEN}✓ Successfully deployed: $DEPLOYED${NC}"
if [ $FAILED -gt 0 ]; then
    echo -e "${RED}✗ Failed: $FAILED${NC}"
fi
echo ""

if [ $DEPLOYED -gt 0 ]; then
    echo -e "${CYAN}Next Steps:${NC}"
    echo "  1. Check deployed functions:"
    echo "     gcloud functions list --project=$PROJECT_ID"
    echo ""
    echo "  2. View function logs:"
    echo "     gcloud functions logs read <function-name> --region=$REGION"
    echo ""
    echo "  3. Check BigQuery tables:"
    echo "     bq ls security_insights"
    echo "     bq ls security_data"
    echo ""
    echo "  4. Test the Security Agent:"
    echo "     python -c \"from agents.agent import root_agent; print(root_agent.chat('List datasets'))\""
    echo ""
fi

exit 0
