# GCP Security Agent - Operations Manual

## 1. Operations Overview

### 1.1 Operational Responsibilities
This manual provides comprehensive guidance for operating, monitoring, and maintaining the GCP Security Agent in production environments. It covers deployment, monitoring, troubleshooting, performance optimization, and disaster recovery procedures.

### 1.2 Service Level Objectives (SLOs)
- **Availability**: 99.5% uptime
- **Response Time**: <2 seconds for asset queries, <5 seconds for security analysis
- **Error Rate**: <1% of requests
- **Recovery Time Objective (RTO)**: 4 hours
- **Recovery Point Objective (RPO)**: 1 hour

### 1.3 Operational Support Structure
- **L1 Support**: Basic monitoring, health checks, routine maintenance
- **L2 Support**: Incident response, troubleshooting, configuration changes
- **L3 Support**: Complex issues, architectural changes, performance optimization

## 2. Deployment Operations

### 2.1 Production Deployment Process

#### 2.1.1 Pre-Deployment Checklist
```bash
# 1. Verify Prerequisites
gcloud auth list
gcloud config get-value project
gcloud services list --enabled

# 2. Validate Configuration
python -m backend.config.validate_config
python -m pytest tests/deployment/ -v

# 3. Check Resource Quotas
gcloud compute project-info describe --format="value(quotas)"
gcloud run services list --region=us-central1

# 4. Backup Current Configuration
gcloud run services describe gcp-security-agent \
  --region=us-central1 \
  --format="export" > backup_config_$(date +%Y%m%d_%H%M%S).yaml
```

#### 2.1.2 Deployment Commands
```bash
# Standard Deployment
python run_backend.py --cloud --project mgm-digitalconcierge

# Blue-Green Deployment
gcloud run deploy gcp-security-agent-staging \
  --image gcr.io/mgm-digitalconcierge/gcp-security-agent:latest \
  --region us-central1 \
  --no-traffic

# Validate staging deployment
curl -f https://gcp-security-agent-staging-<hash>-uc.a.run.app/health

# Switch traffic
gcloud run services update-traffic gcp-security-agent \
  --to-revisions gcp-security-agent-staging=100 \
  --region us-central1
```

#### 2.1.3 Post-Deployment Verification
```bash
# Health Check
curl -f https://your-service.run.app/health

# API Functionality Test
curl -X POST https://your-service.run.app/api/v1/agent/chat \
  -H "Content-Type: application/json" \
  -d '{"query":"health check","user_id":"ops-test","project_id":"mgm-digitalconcierge"}'

# Performance Test
ab -n 100 -c 10 https://your-service.run.app/health

# Monitor initial metrics
gcloud logging read "resource.type=\"cloud_run_revision\" AND resource.labels.service_name=\"gcp-security-agent\"" \
  --limit 50 --format="value(timestamp,severity,textPayload)"
```

### 2.2 Configuration Management

#### 2.2.1 Environment Configuration
```yaml
# Production Environment Variables
production_config:
  ENVIRONMENT: production
  LOG_LEVEL: WARNING
  DEBUG: false
  
  # Performance Settings
  MEMORY_LIMIT: 2Gi
  CPU_LIMIT: 2
  TIMEOUT: 300
  MAX_INSTANCES: 10
  MIN_INSTANCES: 2
  
  # Security Settings
  RATE_LIMIT_ENABLED: true
  RATE_LIMIT_PER_HOUR: 1000
  SESSION_TIMEOUT: 3600
  
  # Monitoring
  ENABLE_MONITORING: true
  ENABLE_TRACING: true
  PERFORMANCE_MONITORING_ENABLED: true
```

#### 2.2.2 Secret Management
```bash
# Store secrets in Google Secret Manager
echo -n "service-account-key-content" | \
  gcloud secrets create security-agent-sa-key --data-file=-

# Update service with secret access
gcloud run services update gcp-security-agent \
  --region=us-central1 \
  --set-env-vars="USE_SECRET_MANAGER=true" \
  --set-env-vars="SERVICE_ACCOUNT_SECRET_NAME=security-agent-sa-key"

# Verify secret access
gcloud secrets versions access latest --secret="security-agent-sa-key"
```

## 3. Monitoring and Alerting

### 3.1 Monitoring Setup

#### 3.1.1 Cloud Monitoring Configuration
```yaml
# monitoring_config.yaml
resources:
  - name: "GCP Security Agent"
    type: "cloud_run_revision"
    filters:
      service_name: "gcp-security-agent"

metrics:
  - name: "Request Rate"
    metric: "run.googleapis.com/request_count"
    threshold: 1000  # requests per minute
    
  - name: "Response Latency"
    metric: "run.googleapis.com/request_latencies"
    threshold: 5000  # milliseconds
    
  - name: "Error Rate"
    metric: "run.googleapis.com/request_count"
    filter: 'response_code_class="4xx" OR response_code_class="5xx"'
    threshold: 50  # errors per minute
    
  - name: "Memory Utilization"
    metric: "run.googleapis.com/container/memory/utilizations"
    threshold: 0.8  # 80%
    
  - name: "CPU Utilization"
    metric: "run.googleapis.com/container/cpu/utilizations"
    threshold: 0.7  # 70%
```

#### 3.1.2 Custom Metrics Implementation
```python
# backend/monitoring/custom_metrics.py
from google.cloud import monitoring_v3
import time

class CustomMetrics:
    def __init__(self, project_id: str):
        self.client = monitoring_v3.MetricServiceClient()
        self.project_name = f"projects/{project_id}"
    
    def record_asset_discovery_count(self, count: int):
        """Record number of assets discovered"""
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/security_agent/asset_discovery_count"
        series.resource.type = "cloud_run_revision"
        
        point = series.points.add()
        point.value.int64_value = count
        point.interval.end_time.seconds = int(time.time())
        
        self.client.create_time_series(
            name=self.project_name,
            time_series=[series]
        )
    
    def record_security_finding_count(self, severity: str, count: int):
        """Record security findings by severity"""
        series = monitoring_v3.TimeSeries()
        series.metric.type = "custom.googleapis.com/security_agent/security_findings"
        series.metric.labels["severity"] = severity
        series.resource.type = "cloud_run_revision"
        
        point = series.points.add()
        point.value.int64_value = count
        point.interval.end_time.seconds = int(time.time())
        
        self.client.create_time_series(
            name=self.project_name,
            time_series=[series]
        )
```

#### 3.1.3 Alerting Policies
```bash
# Create alerting policies
gcloud alpha monitoring policies create --policy-from-file=alerting_policies.yaml

# alerting_policies.yaml content:
```

```yaml
displayName: "GCP Security Agent - High Error Rate"
conditions:
  - displayName: "Error rate > 5%"
    conditionThreshold:
      filter: 'resource.type="cloud_run_revision" resource.label.service_name="gcp-security-agent"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 0.05
      duration: 300s
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_RATE
notificationChannels:
  - projects/mgm-digitalconcierge/notificationChannels/ops-team-email
  - projects/mgm-digitalconcierge/notificationChannels/ops-team-slack

---
displayName: "GCP Security Agent - High Response Time"
conditions:
  - displayName: "95th percentile latency > 5 seconds"
    conditionThreshold:
      filter: 'resource.type="cloud_run_revision" resource.label.service_name="gcp-security-agent"'
      comparison: COMPARISON_GREATER_THAN
      thresholdValue: 5000
      duration: 300s
      aggregations:
        - alignmentPeriod: 60s
          perSeriesAligner: ALIGN_PERCENTILE_95
```

### 3.2 Log Management

#### 3.2.1 Structured Logging Configuration
```python
# backend/logging_config.py
import logging
import json
from datetime import datetime

class StructuredFormatter(logging.Formatter):
    def format(self, record):
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "severity": record.levelname,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add trace context if available
        if hasattr(record, 'trace_id'):
            log_entry["trace_id"] = record.trace_id
        
        # Add user context if available  
        if hasattr(record, 'user_id'):
            log_entry["user_id"] = record.user_id
            
        if hasattr(record, 'session_id'):
            log_entry["session_id"] = record.session_id
        
        return json.dumps(log_entry)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('/var/log/security_agent.log')
    ]
)

# Set custom formatter
for handler in logging.root.handlers:
    handler.setFormatter(StructuredFormatter())
```

#### 3.2.2 Log Analysis Queries
```sql
-- Cloud Logging Queries

-- Error Analysis
SELECT
  timestamp,
  severity,
  jsonPayload.message,
  jsonPayload.user_id,
  jsonPayload.session_id
FROM
  `mgm-digitalconcierge.cloud_logging.gcp_security_agent_logs`
WHERE
  severity >= "ERROR"
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
ORDER BY timestamp DESC

-- Performance Analysis
SELECT
  timestamp,
  jsonPayload.response_time_ms,
  jsonPayload.endpoint,
  jsonPayload.agent_used
FROM
  `mgm-digitalconcierge.cloud_logging.gcp_security_agent_logs`
WHERE
  jsonPayload.response_time_ms > 5000
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 1 HOUR)

-- Security Event Analysis
SELECT
  timestamp,
  jsonPayload.user_id,
  jsonPayload.query,
  jsonPayload.security_findings
FROM
  `mgm-digitalconcierge.cloud_logging.gcp_security_agent_logs`
WHERE
  jsonPayload.security_findings IS NOT NULL
  AND timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 24 HOUR)
```

## 4. Performance Optimization

### 4.1 Performance Monitoring

#### 4.1.1 Key Performance Indicators (KPIs)
```yaml
performance_kpis:
  response_time:
    target: "<2s for asset queries, <5s for analysis"
    measurement: "95th percentile response time"
    
  throughput:
    target: "100+ requests per minute"
    measurement: "Peak sustained throughput"
    
  availability:
    target: "99.5% uptime"
    measurement: "Service availability percentage"
    
  resource_utilization:
    cpu_target: "<70% average"
    memory_target: "<80% average"
    measurement: "Container resource utilization"
```

#### 4.1.2 Performance Profiling
```python
# backend/profiling/performance_profiler.py
import time
import functools
import asyncio
from contextlib import asynccontextmanager

class PerformanceProfiler:
    def __init__(self):
        self.metrics = {}
    
    def time_function(self, func_name: str):
        """Decorator to time function execution"""
        def decorator(func):
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.time()
                try:
                    result = await func(*args, **kwargs)
                    return result
                finally:
                    duration = time.time() - start_time
                    self.record_metric(func_name, duration)
            return wrapper
        return decorator
    
    def record_metric(self, name: str, duration: float):
        """Record performance metric"""
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(duration)
        
        # Keep only recent metrics (last 1000)
        if len(self.metrics[name]) > 1000:
            self.metrics[name] = self.metrics[name][-1000:]
    
    def get_stats(self, name: str) -> dict:
        """Get performance statistics"""
        if name not in self.metrics:
            return {}
        
        values = self.metrics[name]
        return {
            "count": len(values),
            "avg": sum(values) / len(values),
            "min": min(values),
            "max": max(values),
            "p95": sorted(values)[int(len(values) * 0.95)]
        }

# Usage example
profiler = PerformanceProfiler()

@profiler.time_function("asset_discovery")
async def discover_assets(query: str):
    # Asset discovery logic
    pass
```

### 4.2 Optimization Strategies

#### 4.2.1 Caching Optimization
```python
# backend/optimization/cache_optimizer.py
import redis
import json
import hashlib
from typing import Optional, Any

class OptimizedCacheManager:
    def __init__(self, redis_url: str):
        self.redis_client = redis.from_url(redis_url)
        self.hit_rate_threshold = 0.7  # 70% hit rate target
        
    def adaptive_ttl(self, key_pattern: str, base_ttl: int) -> int:
        """Calculate adaptive TTL based on access patterns"""
        hit_rate = self.get_hit_rate(key_pattern)
        
        if hit_rate > self.hit_rate_threshold:
            # High hit rate: increase TTL
            return int(base_ttl * 1.5)
        else:
            # Low hit rate: decrease TTL to save memory
            return int(base_ttl * 0.7)
    
    def intelligent_cache_key(self, query: str, context: dict) -> str:
        """Generate intelligent cache key considering query semantics"""
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Extract semantic components
        semantic_hash = hashlib.md5(
            f"{normalized_query}:{context.get('project_id')}:{context.get('resource_type')}"
            .encode()
        ).hexdigest()[:16]
        
        return f"query:{semantic_hash}"
    
    async def get_with_refresh(self, key: str, refresh_func, ttl: int = 300) -> Any:
        """Get from cache with background refresh"""
        value = self.redis_client.get(key)
        
        if value:
            # Check if near expiration (refresh in background)
            remaining_ttl = self.redis_client.ttl(key)
            if remaining_ttl < ttl * 0.1:  # Refresh when 10% of TTL remains
                asyncio.create_task(self.background_refresh(key, refresh_func, ttl))
            
            return json.loads(value)
        else:
            # Cache miss: fetch and cache
            fresh_value = await refresh_func()
            self.redis_client.setex(key, ttl, json.dumps(fresh_value))
            return fresh_value
    
    async def background_refresh(self, key: str, refresh_func, ttl: int):
        """Refresh cache in background"""
        try:
            fresh_value = await refresh_func()
            self.redis_client.setex(key, ttl, json.dumps(fresh_value))
        except Exception as e:
            logger.warning(f"Background cache refresh failed for {key}: {e}")
```

#### 4.2.2 Query Optimization
```python
# backend/optimization/query_optimizer.py
class QueryOptimizer:
    def __init__(self):
        self.common_patterns = {
            "asset_list": ["show", "list", "get", "what"],
            "security_analysis": ["analyze", "security", "vulnerabilities"],
            "recommendations": ["recommend", "suggest", "improve"]
        }
    
    def optimize_asset_query(self, query: str, project_id: str) -> dict:
        """Optimize asset discovery queries"""
        optimization_hints = {
            "use_cache": True,
            "parallel_fetch": False,
            "resource_filters": [],
            "batch_size": 100
        }
        
        # Analyze query for optimization opportunities
        query_lower = query.lower()
        
        # Specific resource type queries can be optimized
        if "compute" in query_lower or "instance" in query_lower:
            optimization_hints["resource_filters"] = ["compute.googleapis.com/Instance"]
            optimization_hints["parallel_fetch"] = True
            
        elif "storage" in query_lower or "bucket" in query_lower:
            optimization_hints["resource_filters"] = ["storage.googleapis.com/Bucket"]
            optimization_hints["batch_size"] = 50  # Buckets have more metadata
            
        # Complex queries benefit from parallel processing
        if len(query.split()) > 10:
            optimization_hints["parallel_fetch"] = True
            optimization_hints["batch_size"] = 50
        
        return optimization_hints
```

## 5. Troubleshooting Guide

### 5.1 Common Issues and Solutions

#### 5.1.1 Service Unavailable (503) Errors
**Symptoms**: API returns 503 Service Unavailable
**Possible Causes**:
- Cloud Run service down
- GCP API quota exceeded
- Authentication issues

**Diagnostic Steps**:
```bash
# Check service status
gcloud run services describe gcp-security-agent --region=us-central1

# Check logs for errors
gcloud logs read "resource.type=\"cloud_run_revision\"" --limit=50

# Check quota usage
gcloud compute project-info describe --format="table(quotas.metric,quotas.usage,quotas.limit)"

# Test authentication
gcloud auth application-default print-access-token
```

**Resolution**:
```bash
# Restart service
gcloud run services update gcp-security-agent --region=us-central1

# Scale up if needed
gcloud run services update gcp-security-agent \
  --max-instances=20 \
  --region=us-central1

# Request quota increase if needed
gcloud compute project-info describe
```

#### 5.1.2 Slow Response Times
**Symptoms**: Response times > 5 seconds
**Possible Causes**:
- GCP API latency
- Cache misses
- Resource constraints

**Diagnostic Steps**:
```bash
# Check response time metrics
gcloud logging read "resource.type=\"cloud_run_revision\" AND jsonPayload.response_time_ms>5000" \
  --limit=20 --format="value(timestamp,jsonPayload.response_time_ms,jsonPayload.endpoint)"

# Check resource utilization
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com/container"

# Test cache performance
redis-cli --scan --pattern "query:*" | wc -l
redis-cli info memory
```

**Resolution**:
```bash
# Scale up resources
gcloud run services update gcp-security-agent \
  --memory=4Gi \
  --cpu=4 \
  --region=us-central1

# Clear and warm cache
redis-cli flushall
curl -X POST https://your-service.run.app/api/v1/cache/warm

# Optimize queries
# Review and optimize slow queries in application code
```

#### 5.1.3 Authentication Failures
**Symptoms**: 401/403 errors, "Authentication failed"
**Possible Causes**:
- Expired service account keys
- Insufficient IAM permissions
- Secret Manager access issues

**Diagnostic Steps**:
```bash
# Check service account
gcloud iam service-accounts describe gcp-security-agent-sa@mgm-digitalconcierge.iam.gserviceaccount.com

# Check IAM policies
gcloud projects get-iam-policy mgm-digitalconcierge

# Test API access manually
gcloud asset search-all-resources --scope=projects/mgm-digitalconcierge --asset-types=compute.googleapis.com/Instance
```

**Resolution**:
```bash
# Regenerate service account key
gcloud iam service-accounts keys create new-key.json \
  --iam-account=gcp-security-agent-sa@mgm-digitalconcierge.iam.gserviceaccount.com

# Update secret in Secret Manager
gcloud secrets versions add security-agent-sa-key --data-file=new-key.json

# Grant missing permissions
gcloud projects add-iam-policy-binding mgm-digitalconcierge \
  --member="serviceAccount:gcp-security-agent-sa@mgm-digitalconcierge.iam.gserviceaccount.com" \
  --role="roles/cloudasset.viewer"
```

### 5.2 Performance Troubleshooting

#### 5.2.1 Memory Issues
**Symptoms**: Out of Memory errors, high memory usage
**Diagnostic Commands**:
```bash
# Check memory metrics
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com/container/memory"

# Analyze memory usage patterns
gcloud logging read "resource.type=\"cloud_run_revision\" AND severity=\"ERROR\" AND textPayload:\"memory\"" \
  --limit=50
```

**Resolution**:
```bash
# Increase memory limit
gcloud run services update gcp-security-agent \
  --memory=4Gi \
  --region=us-central1

# Optimize memory usage in code
# - Implement proper connection pooling
# - Clear large objects after use
# - Use generators for large datasets
```

#### 5.2.2 CPU Bottlenecks
**Symptoms**: High CPU usage, request queuing
**Diagnostic Commands**:
```bash
# Check CPU metrics
gcloud monitoring metrics list --filter="metric.type:run.googleapis.com/container/cpu"

# Analyze CPU-intensive operations
gcloud logging read "resource.type=\"cloud_run_revision\" AND jsonPayload.cpu_time_ms>1000" \
  --limit=20
```

**Resolution**:
```bash
# Increase CPU allocation
gcloud run services update gcp-security-agent \
  --cpu=4 \
  --region=us-central1

# Implement async processing
# - Use background tasks for heavy operations
# - Implement request queuing
# - Optimize algorithms
```

## 6. Backup and Disaster Recovery

### 6.1 Backup Procedures

#### 6.1.1 Configuration Backup
```bash
#!/bin/bash
# backup_configuration.sh

DATE=$(date +%Y%m%d_%H%M%S)
BACKUP_DIR="/backups/gcp-security-agent"
BUCKET="gs://mgm-digitalconcierge-backups"

# Create backup directory
mkdir -p $BACKUP_DIR

# Backup Cloud Run configuration
gcloud run services describe gcp-security-agent \
  --region=us-central1 \
  --format="export" > $BACKUP_DIR/cloud_run_config_$DATE.yaml

# Backup IAM policies
gcloud projects get-iam-policy mgm-digitalconcierge \
  --format="export" > $BACKUP_DIR/iam_policy_$DATE.yaml

# Backup secrets list
gcloud secrets list --format="value(name)" > $BACKUP_DIR/secrets_list_$DATE.txt

# Backup monitoring policies
gcloud alpha monitoring policies list \
  --format="export" > $BACKUP_DIR/monitoring_policies_$DATE.yaml

# Upload to Cloud Storage
gsutil -m cp -r $BACKUP_DIR/* $BUCKET/configuration/

# Clean old local backups (keep 7 days)
find $BACKUP_DIR -name "*.yaml" -mtime +7 -delete
find $BACKUP_DIR -name "*.txt" -mtime +7 -delete

echo "Backup completed: $DATE"
```

#### 6.1.2 Session Data Backup
```bash
#!/bin/bash
# backup_session_data.sh

DATE=$(date +%Y%m%d_%H%M%S)
REDIS_HOST="your-redis-host"
BUCKET="gs://mgm-digitalconcierge-backups"

# Backup Redis data
redis-cli --rdb /tmp/redis_backup_$DATE.rdb
gzip /tmp/redis_backup_$DATE.rdb

# Upload to Cloud Storage
gsutil cp /tmp/redis_backup_$DATE.rdb.gz $BUCKET/session-data/

# Clean old backups (keep 30 days)
gsutil -m rm $BUCKET/session-data/redis_backup_$(date -d '30 days ago' +%Y%m%d)*.rdb.gz

# Clean local backup
rm /tmp/redis_backup_$DATE.rdb.gz

echo "Session data backup completed: $DATE"
```

### 6.2 Disaster Recovery Procedures

#### 6.2.1 Service Recovery
```bash
#!/bin/bash
# disaster_recovery.sh

REGION="us-central1"
SERVICE_NAME="gcp-security-agent"
BACKUP_DATE="20240101_120000"  # Latest backup date

echo "Starting disaster recovery procedure..."

# Step 1: Deploy service from backup configuration
gsutil cp gs://mgm-digitalconcierge-backups/configuration/cloud_run_config_$BACKUP_DATE.yaml ./recovery_config.yaml

# Step 2: Restore IAM policies
gsutil cp gs://mgm-digitalconcierge-backups/configuration/iam_policy_$BACKUP_DATE.yaml ./iam_policy.yaml
gcloud projects set-iam-policy mgm-digitalconcierge iam_policy.yaml

# Step 3: Deploy service
gcloud run services replace recovery_config.yaml --region=$REGION

# Step 4: Restore session data
gsutil cp gs://mgm-digitalconcierge-backups/session-data/redis_backup_$BACKUP_DATE.rdb.gz ./
gunzip redis_backup_$BACKUP_DATE.rdb.gz
redis-cli --rdb redis_backup_$BACKUP_DATE.rdb

# Step 5: Verify service health
curl -f https://$(gcloud run services describe $SERVICE_NAME --region=$REGION --format="value(status.url)")/health

# Step 6: Run smoke tests
python -m pytest tests/smoke/ -v

echo "Disaster recovery completed. Please verify all functionality."
```

#### 6.2.2 Data Recovery Verification
```python
# scripts/verify_recovery.py
import requests
import json

def verify_recovery(service_url: str):
    """Verify service recovery by testing key functionality"""
    
    test_cases = [
        {
            "name": "Health Check",
            "endpoint": "/health",
            "method": "GET",
            "expected_status": 200
        },
        {
            "name": "Asset Discovery",
            "endpoint": "/api/v1/asset-inventory/summary",
            "method": "GET",
            "expected_status": 200
        },
        {
            "name": "Chat Interface",
            "endpoint": "/api/v1/agent/chat",
            "method": "POST",
            "data": {
                "query": "test recovery",
                "user_id": "recovery-test",
                "project_id": "mgm-digitalconcierge"
            },
            "expected_status": 200
        }
    ]
    
    results = []
    for test in test_cases:
        try:
            if test["method"] == "GET":
                response = requests.get(f"{service_url}{test['endpoint']}")
            else:
                response = requests.post(
                    f"{service_url}{test['endpoint']}", 
                    json=test.get("data")
                )
            
            success = response.status_code == test["expected_status"]
            results.append({
                "test": test["name"],
                "success": success,
                "status_code": response.status_code,
                "response_time": response.elapsed.total_seconds()
            })
            
        except Exception as e:
            results.append({
                "test": test["name"],
                "success": False,
                "error": str(e)
            })
    
    return results

if __name__ == "__main__":
    service_url = "https://your-service.run.app"
    results = verify_recovery(service_url)
    
    print("Recovery Verification Results:")
    for result in results:
        status = "✅" if result["success"] else "❌"
        print(f"{status} {result['test']}: {result.get('status_code', 'ERROR')}")
    
    success_rate = sum(1 for r in results if r["success"]) / len(results)
    print(f"\nOverall Success Rate: {success_rate:.1%}")
```

## 7. Security Operations

### 7.1 Security Monitoring

#### 7.1.1 Security Event Detection
```python
# backend/security/security_monitor.py
import re
from datetime import datetime
from typing import List, Dict

class SecurityEventMonitor:
    def __init__(self):
        self.security_patterns = {
            "injection_attempt": [
                r"(union|select|insert|update|delete|drop)\s",
                r"<script[^>]*>.*?</script>",
                r"javascript:",
                r"\$\([^)]*\)",
            ],
            "path_traversal": [
                r"\.\./",
                r"\.\.\\",
                r"%2e%2e%2f",
                r"%2e%2e%5c"
            ],
            "suspicious_commands": [
                r";\s*(ls|cat|rm|curl|wget|chmod)",
                r"\|\s*(cat|grep|awk|sed)",
                r"&&\s*(rm|curl|wget)"
            ]
        }
    
    def analyze_request(self, request_data: dict) -> Dict[str, any]:
        """Analyze request for security threats"""
        security_events = []
        risk_level = "LOW"
        
        query = request_data.get("query", "")
        user_id = request_data.get("user_id", "")
        
        # Check for injection patterns
        for category, patterns in self.security_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    security_events.append({
                        "type": category,
                        "pattern": pattern,
                        "matched_text": re.search(pattern, query, re.IGNORECASE).group(),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    risk_level = "HIGH"
        
        # Check for unusual patterns
        if len(query) > 1000:
            security_events.append({
                "type": "large_payload",
                "size": len(query),
                "timestamp": datetime.utcnow().isoformat()
            })
            risk_level = max(risk_level, "MEDIUM")
        
        return {
            "risk_level": risk_level,
            "security_events": security_events,
            "user_id": user_id,
            "timestamp": datetime.utcnow().isoformat()
        }
```

#### 7.1.2 Automated Security Response
```python
# backend/security/auto_response.py
class AutoSecurityResponse:
    def __init__(self):
        self.blocked_users = set()
        self.rate_limits = {}
    
    async def handle_security_event(self, event: dict):
        """Handle detected security events"""
        risk_level = event["risk_level"]
        user_id = event["user_id"]
        
        if risk_level == "HIGH":
            # Block user temporarily
            self.blocked_users.add(user_id)
            
            # Send alert
            await self.send_security_alert(event)
            
            # Log to security log
            await self.log_security_event(event)
            
        elif risk_level == "MEDIUM":
            # Apply rate limiting
            self.apply_rate_limit(user_id)
            
        # Always log security events
        await self.log_security_event(event)
    
    async def send_security_alert(self, event: dict):
        """Send security alert to operations team"""
        alert_message = {
            "alert_type": "SECURITY_INCIDENT",
            "severity": event["risk_level"], 
            "user_id": event["user_id"],
            "events": event["security_events"],
            "timestamp": event["timestamp"],
            "action_taken": "USER_BLOCKED" if event["risk_level"] == "HIGH" else "RATE_LIMITED"
        }
        
        # Send to monitoring system
        # Implementation depends on alerting system
        print(f"SECURITY ALERT: {alert_message}")
```

### 7.2 Compliance Operations

#### 7.2.1 Compliance Monitoring
```yaml
# compliance_config.yaml
compliance_frameworks:
  SOC2:
    requirements:
      - "Access control and authentication"
      - "Data encryption in transit and at rest"
      - "Audit logging and monitoring"
      - "Incident response procedures"
    monitoring:
      - "Authentication failure rates"
      - "Data access patterns"
      - "Security event frequencies"
      
  ISO27001:
    requirements:
      - "Information security management system"
      - "Risk assessment and treatment"
      - "Security incident management"
      - "Business continuity planning"
    monitoring:
      - "Security control effectiveness"
      - "Risk mitigation status"
      - "Incident response times"
      
  GDPR:
    requirements:
      - "Data protection by design"
      - "Data subject rights"
      - "Data breach notification"
      - "Privacy impact assessments"
    monitoring:
      - "Personal data processing"
      - "Data retention compliance"
      - "Consent management"
```

#### 7.2.2 Compliance Reporting
```python
# scripts/compliance_report.py
from datetime import datetime, timedelta
import json

class ComplianceReporter:
    def __init__(self):
        self.frameworks = ["SOC2", "ISO27001", "GDPR"]
    
    def generate_compliance_report(self, timeframe_days: int = 30) -> dict:
        """Generate compliance report for specified timeframe"""
        end_date = datetime.utcnow()
        start_date = end_date - timedelta(days=timeframe_days)
        
        report = {
            "report_period": {
                "start": start_date.isoformat(),
                "end": end_date.isoformat()
            },
            "frameworks": {}
        }
        
        for framework in self.frameworks:
            report["frameworks"][framework] = self.assess_framework_compliance(
                framework, start_date, end_date
            )
        
        return report
    
    def assess_framework_compliance(self, framework: str, start_date: datetime, end_date: datetime) -> dict:
        """Assess compliance for specific framework"""
        # Implementation would check actual compliance metrics
        # This is a template showing the structure
        
        if framework == "SOC2":
            return {
                "overall_score": 95.5,
                "controls": {
                    "access_control": {"score": 98, "status": "COMPLIANT"},
                    "data_encryption": {"score": 95, "status": "COMPLIANT"},
                    "audit_logging": {"score": 92, "status": "COMPLIANT"},
                    "incident_response": {"score": 97, "status": "COMPLIANT"}
                },
                "findings": [
                    {
                        "severity": "LOW",
                        "description": "Minor logging delay observed",
                        "remediation": "Increase log buffer size"
                    }
                ]
            }
        
        return {"overall_score": 0, "controls": {}, "findings": []}

if __name__ == "__main__":
    reporter = ComplianceReporter()
    report = reporter.generate_compliance_report(30)
    
    print("Compliance Report Generated:")
    print(json.dumps(report, indent=2))
```

This operations manual provides comprehensive guidance for maintaining the GCP Security Agent in production, covering monitoring, troubleshooting, performance optimization, disaster recovery, and security operations. Regular review and updates of these procedures ensure continued operational excellence.