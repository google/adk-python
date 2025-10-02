#!/bin/bash

# Security Agent System Status Check
# Comprehensive monitoring of all deployed components

echo "======================================================================"
echo "🔍 SECURITY AGENT SYSTEM STATUS CHECK"
echo "======================================================================"
echo "Time: $(date)"
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to check service status
check_service() {
    local service_name=$1
    local check_command=$2

    echo -n "Checking $service_name... "

    if eval $check_command > /dev/null 2>&1; then
        echo -e "${GREEN}✅ Running${NC}"
        return 0
    else
        echo -e "${RED}❌ Not Running${NC}"
        return 1
    fi
}

# Function to check URL endpoint
check_endpoint() {
    local endpoint_name=$1
    local url=$2

    echo -n "Checking $endpoint_name... "

    response=$(curl -s -o /dev/null -w "%{http_code}" "$url" 2>/dev/null)

    if [ "$response" = "200" ]; then
        echo -e "${GREEN}✅ OK (200)${NC}"
        return 0
    elif [ "$response" = "000" ]; then
        echo -e "${RED}❌ Not Responding${NC}"
        return 1
    else
        echo -e "${YELLOW}⚠️ Status: $response${NC}"
        return 1
    fi
}

echo "1️⃣ LOCAL SERVICES"
echo "----------------------------------------------------------------------"

# Check ADK Agent
check_service "ADK Agent (port 8000)" "lsof -i :8000 | grep LISTEN"

# Check Flask API
check_service "Flask API (port 5000)" "lsof -i :5000 | grep LISTEN"

# Check Streamlit Frontend
check_service "Streamlit Frontend (port 8501)" "lsof -i :8501 | grep LISTEN"

echo ""
echo "2️⃣ API ENDPOINTS"
echo "----------------------------------------------------------------------"

# Check API endpoints
check_endpoint "ADK Health" "http://localhost:8000/health"
check_endpoint "Flask Health" "http://localhost:5000/health"
check_endpoint "Service Discovery API" "http://localhost:5000/api/services/categories"
check_endpoint "Metrics API" "http://localhost:5000/api/metrics"

echo ""
echo "3️⃣ CLOUD FUNCTIONS"
echo "----------------------------------------------------------------------"

# List deployed Cloud Functions
echo "Deployed Functions:"
gcloud functions list --format="table(name,status,trigger)" 2>/dev/null | head -15

echo ""
echo "4️⃣ BIGQUERY DATASETS"
echo "----------------------------------------------------------------------"

# Check BigQuery datasets
echo "Checking BigQuery datasets..."
bq ls 2>/dev/null | grep -E "(security_insights|learned_services)" || echo "No security datasets found"

echo ""
echo "5️⃣ RECENT CLOUD FUNCTION EXECUTIONS"
echo "----------------------------------------------------------------------"

# Check recent function executions
echo "Recent executions (last 5):"
gcloud functions logs read --limit=5 --format="table(time,function,level,text)" 2>/dev/null | head -10

echo ""
echo "6️⃣ CLOUD MONITORING METRICS"
echo "----------------------------------------------------------------------"

# Check if custom metrics exist
echo "Checking custom metrics..."
gcloud monitoring metrics-descriptors list --filter="metric.type:custom.googleapis.com" --limit=5 2>/dev/null | grep -E "(security|agent|confluence|url_learning)" || echo "No custom metrics found"

echo ""
echo "7️⃣ ALERT POLICIES"
echo "----------------------------------------------------------------------"

# List alert policies
echo "Active alert policies:"
gcloud alpha monitoring policies list --format="table(displayName,enabled)" 2>/dev/null | head -10

echo ""
echo "8️⃣ SYSTEM RESOURCES"
echo "----------------------------------------------------------------------"

# Check disk space
echo -n "Disk Space: "
df -h . | tail -1 | awk '{print $4 " available (" $5 " used)"}'

# Check memory
echo -n "Memory: "
if [[ "$OSTYPE" == "darwin"* ]]; then
    vm_stat | perl -ne '/page size of (\d+)/ and $size=$1; /Pages free:\s+(\d+)/ and printf("%.1f GB free\n", $1*$size/1073741824);'
else
    free -h | grep Mem | awk '{print $4 " free"}'
fi

# Check Python processes
echo -n "Python Processes: "
ps aux | grep python | grep -v grep | wc -l

echo ""
echo "9️⃣ CACHE & DATA STATUS"
echo "----------------------------------------------------------------------"

# Check cache files
echo "Cache files:"
ls -lh cache/*.db 2>/dev/null | tail -5 || echo "No cache databases found"

echo ""
echo "🔟 URL LEARNING STATUS"
echo "----------------------------------------------------------------------"

# Check learned services
if [ -f "cache/service_docs/parsed_services.db" ]; then
    echo "Learned services:"
    sqlite3 cache/service_docs/parsed_services.db "SELECT service_name, url, parse_date FROM parsed_services ORDER BY parse_date DESC LIMIT 5;" 2>/dev/null || echo "Unable to query learned services"
else
    echo "No learned services database found"
fi

echo ""
echo "======================================================================"
echo "📊 SUMMARY"
echo "----------------------------------------------------------------------"

# Count successes and failures
services_up=0
services_down=0

# Recheck services for summary
lsof -i :8000 | grep LISTEN > /dev/null 2>&1 && ((services_up++)) || ((services_down++))
lsof -i :5000 | grep LISTEN > /dev/null 2>&1 && ((services_up++)) || ((services_down++))
lsof -i :8501 | grep LISTEN > /dev/null 2>&1 && ((services_up++)) || ((services_down++))

echo "Local Services: $services_up UP / $services_down DOWN"

# Check if monitoring is set up
if gcloud monitoring metrics-descriptors list --filter="metric.type:custom.googleapis.com" 2>/dev/null | grep -q "custom.googleapis.com"; then
    echo -e "Cloud Monitoring: ${GREEN}✅ Configured${NC}"
else
    echo -e "Cloud Monitoring: ${YELLOW}⚠️ Not Configured${NC}"
fi

# Overall status
if [ $services_down -eq 0 ]; then
    echo -e "\nOverall Status: ${GREEN}✅ HEALTHY${NC}"
elif [ $services_up -gt 0 ]; then
    echo -e "\nOverall Status: ${YELLOW}⚠️ PARTIALLY RUNNING${NC}"
else
    echo -e "\nOverall Status: ${RED}❌ SYSTEM DOWN${NC}"
fi

echo ""
echo "======================================================================"
echo "📝 MONITORING URLS"
echo "----------------------------------------------------------------------"
echo "Dashboard: https://console.cloud.google.com/monitoring/dashboards"
echo "Alerts: https://console.cloud.google.com/monitoring/alerting/policies"
echo "Logs: https://console.cloud.google.com/logs"
echo "Functions: https://console.cloud.google.com/functions"
echo "BigQuery: https://console.cloud.google.com/bigquery"
echo "======================================================================