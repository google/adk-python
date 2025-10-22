#!/bin/bash
# Docker Preflight Check - Validates Docker prerequisites before build/run
# Usage: ./scripts/docker_preflight.sh

set -e

# Colors for output
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  Docker Preflight Check${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Track validation status
VALIDATION_PASSED=true

# Check 1: config directory exists
echo -e "${BLUE}[1/5] Checking config directory...${NC}"
if [ -d "config" ]; then
    echo -e "${GREEN}✓${NC} config/ directory exists"
else
    echo -e "${RED}✗${NC} config/ directory missing"
    echo -e "  ${YELLOW}→${NC} Creating config/ directory..."
    mkdir -p config
    echo -e "${GREEN}✓${NC} config/ directory created"
fi
echo ""

# Check 2: .env file exists
echo -e "${BLUE}[2/5] Checking .env file...${NC}"
if [ -f ".env" ]; then
    echo -e "${GREEN}✓${NC} .env file exists"

    # Validate required variables
    required_vars=("GOOGLE_CLOUD_PROJECT" "GOOGLE_APPLICATION_CREDENTIALS" "BQ_DEFAULT_DATASET" "BQ_DEFAULT_TABLE")
    missing_vars=()

    for var in "${required_vars[@]}"; do
        if ! grep -q "^${var}=" .env 2>/dev/null || grep -q "^${var}=$" .env 2>/dev/null || grep -q "^${var}=your-" .env 2>/dev/null; then
            missing_vars+=("$var")
        fi
    done

    if [ ${#missing_vars[@]} -eq 0 ]; then
        echo -e "${GREEN}✓${NC} All required variables configured"
    else
        echo -e "${YELLOW}⚠${NC}  Required variables not configured: ${missing_vars[*]}"
        echo -e "  ${YELLOW}→${NC} Edit .env and set these variables"
        VALIDATION_PASSED=false
    fi
else
    echo -e "${RED}✗${NC} .env file not found"
    echo -e "  ${YELLOW}→${NC} Copy .env.example to .env:"
    echo -e "  ${BLUE}cp .env.example .env${NC}"
    echo -e "  ${YELLOW}→${NC} Then edit .env with your GCP project details"
    VALIDATION_PASSED=false
fi
echo ""

# Check 3: Service account JSON exists
echo -e "${BLUE}[3/5] Checking service account credentials...${NC}"

# Try to extract the path from .env
if [ -f ".env" ]; then
    SA_PATH=$(grep "^GOOGLE_APPLICATION_CREDENTIALS=" .env 2>/dev/null | cut -d= -f2 | tr -d '"' | tr -d "'")

    if [ -z "$SA_PATH" ]; then
        echo -e "${YELLOW}⚠${NC}  GOOGLE_APPLICATION_CREDENTIALS not set in .env"
        echo -e "  ${YELLOW}→${NC} Set it to: config/service-account-key.json"
        VALIDATION_PASSED=false
    elif [ -f "$SA_PATH" ]; then
        echo -e "${GREEN}✓${NC} Service account file exists: $SA_PATH"

        # Check file permissions (should be 600 or 400)
        PERMS=$(stat -c "%a" "$SA_PATH" 2>/dev/null || stat -f "%Lp" "$SA_PATH" 2>/dev/null)
        if [ "$PERMS" = "600" ] || [ "$PERMS" = "400" ]; then
            echo -e "${GREEN}✓${NC} File permissions are secure: $PERMS"
        else
            echo -e "${YELLOW}⚠${NC}  File permissions too open: $PERMS"
            echo -e "  ${YELLOW}→${NC} Secure with: chmod 600 $SA_PATH"
        fi
    else
        echo -e "${YELLOW}⚠${NC}  Service account file not found: $SA_PATH"
        echo -e "  ${YELLOW}→${NC} Place your GCP service account JSON at: $SA_PATH"
        echo -e "  ${YELLOW}→${NC} See config/README.md for detailed setup instructions"
        VALIDATION_PASSED=false
    fi
else
    echo -e "${YELLOW}⚠${NC}  Cannot check (no .env file)"
    VALIDATION_PASSED=false
fi
echo ""

# Check 4: Docker is installed
echo -e "${BLUE}[4/5] Checking Docker installation...${NC}"
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version | cut -d' ' -f3 | sed 's/,//')
    echo -e "${GREEN}✓${NC} Docker is installed: $DOCKER_VERSION"

    # Check if Docker daemon is running
    if docker info &> /dev/null; then
        echo -e "${GREEN}✓${NC} Docker daemon is running"
    else
        echo -e "${RED}✗${NC} Docker daemon is not running"
        echo -e "  ${YELLOW}→${NC} Start Docker Desktop or Docker service"
        VALIDATION_PASSED=false
    fi
else
    echo -e "${RED}✗${NC} Docker is not installed"
    echo -e "  ${YELLOW}→${NC} Install from: https://docs.docker.com/get-docker/"
    VALIDATION_PASSED=false
fi
echo ""

# Check 5: Docker Compose is available
echo -e "${BLUE}[5/5] Checking Docker Compose...${NC}"
if docker compose version &> /dev/null; then
    COMPOSE_VERSION=$(docker compose version --short 2>/dev/null || echo "unknown")
    echo -e "${GREEN}✓${NC} Docker Compose is available: $COMPOSE_VERSION"
elif command -v docker-compose &> /dev/null; then
    COMPOSE_VERSION=$(docker-compose --version | cut -d' ' -f3 | sed 's/,//')
    echo -e "${GREEN}✓${NC} Docker Compose is available: $COMPOSE_VERSION"
    echo -e "${YELLOW}⚠${NC}  Using legacy docker-compose command"
else
    echo -e "${YELLOW}⚠${NC}  Docker Compose not found (optional)"
    echo -e "  ${BLUE}→${NC} Can still use ./scripts/docker_build.sh and docker_run.sh"
fi
echo ""

# Summary
echo -e "${BLUE}========================================${NC}"
if [ "$VALIDATION_PASSED" = true ]; then
    echo -e "${GREEN}✓ All checks passed!${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${GREEN}Ready to build and run:${NC}"
    echo -e "  ${BLUE}→${NC} docker compose up --build"
    echo -e "  ${BLUE}→${NC} ./scripts/docker_build.sh && ./scripts/docker_run.sh"
    echo ""
    exit 0
else
    echo -e "${RED}✗ Validation failed${NC}"
    echo -e "${BLUE}========================================${NC}"
    echo ""
    echo -e "${YELLOW}Fix the issues above before running Docker${NC}"
    echo ""
    echo -e "${BLUE}Quick Setup Guide:${NC}"
    echo -e "  ${GREEN}1.${NC} cp .env.example .env"
    echo -e "  ${GREEN}2.${NC} Edit .env with your GCP project ID"
    echo -e "  ${GREEN}3.${NC} Place service account JSON in config/service-account-key.json"
    echo -e "  ${GREEN}4.${NC} Run this script again to verify"
    echo ""
    echo -e "For detailed instructions, see: ${BLUE}config/README.md${NC}"
    echo ""
    exit 1
fi
