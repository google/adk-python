#!/bin/bash

# Check status of ADK Security Agent services

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo "🔍 ADK Security Agent Status"
echo "============================="

# Check backend
if curl -s http://localhost:8000/list-apps > /dev/null 2>&1; then
    echo -e "Backend:  ${GREEN}✅ Running${NC} (http://localhost:8000)"
else
    echo -e "Backend:  ${RED}❌ Not running${NC}"
fi

# Check frontend
if curl -s -I http://localhost:8501 > /dev/null 2>&1; then
    echo -e "Frontend: ${GREEN}✅ Running${NC} (http://localhost:8501)"
else
    echo -e "Frontend: ${RED}❌ Not running${NC}"
fi

# Check database
if [ -f "backend/cache/gcp_data.db" ]; then
    SIZE=$(du -h backend/cache/gcp_data.db | cut -f1)
    echo -e "Database: ${GREEN}✅ Exists${NC} (Size: $SIZE)"
else
    echo -e "Database: ${RED}❌ Not found${NC}"
fi

# Check for duplicate processes
echo -e "\n${YELLOW}Process Check:${NC}"
ADK_COUNT=$(pgrep -f "adk web" | wc -l)
STREAMLIT_COUNT=$(pgrep -f "streamlit" | wc -l)

if [ $ADK_COUNT -gt 1 ]; then
    echo -e "${RED}⚠️  Warning: $ADK_COUNT ADK processes running (should be 1)${NC}"
elif [ $ADK_COUNT -eq 1 ]; then
    echo -e "${GREEN}✅ Single ADK backend process${NC}"
else
    echo -e "${YELLOW}⚠️  No ADK backend process${NC}"
fi

if [ $STREAMLIT_COUNT -gt 1 ]; then
    echo -e "${RED}⚠️  Warning: $STREAMLIT_COUNT Streamlit processes running (should be 1)${NC}"
elif [ $STREAMLIT_COUNT -eq 1 ]; then
    echo -e "${GREEN}✅ Single Streamlit frontend process${NC}"
else
    echo -e "${YELLOW}⚠️  No Streamlit frontend process${NC}"
fi