"""Security Analytics service with BigQuery integration."""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import json
import os
from google.cloud import bigquery
from google.auth import default

# OpenTelemetry imports
from opentelemetry import trace
from opentelemetry.trace import Status, StatusCode

from .models import (
    SecurityAnalyticsRequest, SecurityAnalyticsResponse, SecurityEvent,
    SecurityAnomaly, ThreatIntelligence, SecurityMetrics, ComplianceViolation,
    SecurityTrend, RiskAssessment, QueryTemplate, SecurityDashboard,
    SecurityEventSeverity, AnomalyType
)

logger = logging.getLogger(__name__)
tracer = trace.get_tracer(__name__)


class SecurityAnalyticsService:
    """Advanced security analytics using BigQuery."""
    
    def __init__(self, project_id: str = None):
        """Initialize the security analytics service."""
        self.project_id = project_id
        self.bq_client = None
        
        # Service configuration flags
        self.enabled = True  # Master switch for the service
        self.use_bigquery = os.getenv("ENABLE_BIGQUERY_ANALYTICS", "false").lower() == "true"
        self.bigquery_initialized = False
        
        self.query_templates = self._load_query_templates()
        self.anomaly_thresholds = self._load_anomaly_thresholds()
        
    async def initialize(self):
        """Initialize BigQuery client - non-blocking."""
        try:
            if not self.use_bigquery:
                logger.info("⚠️ BigQuery analytics disabled by configuration")
                return False
                
            credentials, project = default()
            self.project_id = self.project_id or project
            self.bq_client = bigquery.Client(
                project=self.project_id,
                credentials=credentials
            )
            self.bigquery_initialized = True
            logger.info(f"✅ Initialized BigQuery client for project: {self.project_id}")
            return True
            
        except Exception as e:
            logger.warning(f"⚠️ BigQuery not available: {e}")
            logger.info("✅ Falling back to sample analytics mode")
            self.bigquery_initialized = False
            self.use_bigquery = False
            return False
    
    async def run_security_analytics(self, request: SecurityAnalyticsRequest) -> SecurityAnalyticsResponse:
        """Run security analytics query."""
        start_time = datetime.utcnow()
        
        # Check if service is enabled
        if not self.enabled:
            return SecurityAnalyticsResponse(
                success=False,
                query_type=request.query_type,
                project_id=request.project_id,
                execution_time_ms=0,
                results_count=0,
                error="Security Analytics service is disabled"
            )
        
        with tracer.start_as_current_span("security_analytics_query") as span:
            span.set_attribute("query_type", request.query_type)
            span.set_attribute("project_id", request.project_id)
            span.set_attribute("time_range_hours", request.time_range_hours)
            
            try:
                # If BigQuery is not available, return sample data
                if not self.use_bigquery or not self.bigquery_initialized:
                    if not self.bq_client:
                        await self.initialize()
                    
                    if not self.bigquery_initialized:
                        # Return sample data instead of failing
                        processed_results = await self._generate_sample_results(request)
                        execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                        
                        return SecurityAnalyticsResponse(
                            success=True,
                            query_type=request.query_type,
                            project_id=request.project_id,
                            execution_time_ms=execution_time,
                            results_count=len(processed_results.get("events", [])),
                            **processed_results
                        )
                
                # Get query template
                template = self.query_templates.get(request.query_type)
                if not template:
                    raise ValueError(f"Unknown query type: {request.query_type}")
                
                # Build and execute query
                query = self._build_query(template, request)
                results = await self._execute_query(query)
                
                # Process results based on query type
                processed_results = await self._process_results(
                    request.query_type, results, request
                )
                
                # Calculate execution time
                execution_time = int((datetime.utcnow() - start_time).total_seconds() * 1000)
                
                response = SecurityAnalyticsResponse(
                    success=True,
                    query_type=request.query_type,
                    project_id=request.project_id,
                    execution_time_ms=execution_time,
                    results_count=len(results),
                    **processed_results
                )
                
                if request.include_raw_data:
                    response.raw_query = query
                    response.raw_results = results
                
                span.set_status(Status(StatusCode.OK))
                return response
                
            except Exception as e:
                logger.error(f"Error in security analytics: {e}", exc_info=True)
                span.record_exception(e)
                span.set_status(Status(StatusCode.ERROR, str(e)))
                
                return SecurityAnalyticsResponse(
                    success=False,
                    query_type=request.query_type,
                    project_id=request.project_id,
                    execution_time_ms=int((datetime.utcnow() - start_time).total_seconds() * 1000),
                    results_count=0,
                    error=str(e)
                )
    
    def _load_query_templates(self) -> Dict[str, QueryTemplate]:
        """Load pre-defined security query templates."""
        templates = {
            "anomaly_detection": QueryTemplate(
                template_id="anomaly_detection",
                name="User Behavior Anomaly Detection",
                description="Detect unusual user activity patterns",
                category="behavioral_analysis",
                query_sql="""
                WITH user_baseline AS (
                  SELECT 
                    user_email,
                    AVG(daily_actions) as avg_daily_actions,
                    STDDEV(daily_actions) as stddev_daily_actions,
                    AVG(unique_resources) as avg_unique_resources,
                    STDDEV(unique_resources) as stddev_unique_resources
                  FROM (
                    SELECT 
                      user_email,
                      DATE(timestamp) as date,
                      COUNT(*) as daily_actions,
                      COUNT(DISTINCT resource_name) as unique_resources
                    FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_activity`
                    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {baseline_days} DAY)
                      AND timestamp < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                      AND user_email IS NOT NULL
                    GROUP BY user_email, date
                  ) daily_stats
                  GROUP BY user_email
                  HAVING COUNT(*) >= 7  -- At least 7 days of data
                ),
                recent_activity AS (
                  SELECT 
                    user_email,
                    COUNT(*) as recent_actions,
                    COUNT(DISTINCT resource_name) as recent_unique_resources,
                    ARRAY_AGG(DISTINCT resource_name LIMIT 10) as accessed_resources,
                    MIN(timestamp) as first_activity,
                    MAX(timestamp) as last_activity
                  FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_activity`
                  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                    AND user_email IS NOT NULL
                  GROUP BY user_email
                )
                SELECT 
                  r.user_email,
                  r.recent_actions,
                  r.recent_unique_resources,
                  r.accessed_resources,
                  r.first_activity,
                  r.last_activity,
                  b.avg_daily_actions,
                  b.avg_unique_resources,
                  -- Calculate anomaly scores
                  CASE 
                    WHEN b.stddev_daily_actions > 0 
                    THEN ABS(r.recent_actions - b.avg_daily_actions) / b.stddev_daily_actions
                    ELSE 0 
                  END as actions_z_score,
                  CASE 
                    WHEN b.stddev_unique_resources > 0 
                    THEN ABS(r.recent_unique_resources - b.avg_unique_resources) / b.stddev_unique_resources
                    ELSE 0 
                  END as resources_z_score,
                  -- Determine severity
                  CASE 
                    WHEN (r.recent_actions > b.avg_daily_actions * 3 
                          OR r.recent_unique_resources > b.avg_unique_resources * 2)
                    THEN 'high'
                    WHEN (r.recent_actions > b.avg_daily_actions * 2 
                          OR r.recent_unique_resources > b.avg_unique_resources * 1.5)
                    THEN 'medium'
                    ELSE 'low'
                  END as severity
                FROM recent_activity r
                JOIN user_baseline b ON r.user_email = b.user_email
                WHERE (r.recent_actions > b.avg_daily_actions * 1.5 
                       OR r.recent_unique_resources > b.avg_unique_resources * 1.2)
                ORDER BY 
                  GREATEST(
                    r.recent_actions / NULLIF(b.avg_daily_actions, 0),
                    r.recent_unique_resources / NULLIF(b.avg_unique_resources, 0)
                  ) DESC
                """,
                parameters=["project_id", "baseline_days", "analysis_hours"],
                severity_mapping={
                    "low": SecurityEventSeverity.LOW,
                    "medium": SecurityEventSeverity.MEDIUM, 
                    "high": SecurityEventSeverity.HIGH
                }
            ),
            
            "privilege_escalation": QueryTemplate(
                template_id="privilege_escalation",
                name="Privilege Escalation Detection",
                description="Detect privilege escalation events",
                category="access_control",
                query_sql="""
                WITH iam_changes AS (
                  SELECT 
                    timestamp,
                    user_email,
                    resource_name,
                    method_name,
                    JSON_EXTRACT_SCALAR(request, '$.policy.bindings') as new_bindings,
                    LAG(JSON_EXTRACT_SCALAR(request, '$.policy.bindings')) OVER (
                      PARTITION BY resource_name 
                      ORDER BY timestamp
                    ) as old_bindings
                  FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_activity`
                  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                    AND method_name IN ('SetIamPolicy', 'google.iam.admin.v1.SetIamPolicy')
                    AND user_email IS NOT NULL
                ),
                role_changes AS (
                  SELECT 
                    timestamp,
                    user_email,
                    resource_name,
                    method_name,
                    -- Extract roles from bindings (simplified)
                    REGEXP_EXTRACT_ALL(new_bindings, r'"role":"([^"]*)"') as new_roles,
                    REGEXP_EXTRACT_ALL(old_bindings, r'"role":"([^"]*)"') as old_roles
                  FROM iam_changes
                  WHERE old_bindings IS NOT NULL
                )
                SELECT 
                  timestamp,
                  user_email,
                  resource_name,
                  new_roles,
                  old_roles,
                  -- Check for high-privilege roles
                  CASE 
                    WHEN EXISTS(
                      SELECT 1 FROM UNNEST(new_roles) as role 
                      WHERE role IN ('roles/owner', 'roles/editor', 'roles/iam.securityAdmin')
                    ) THEN 'critical'
                    WHEN EXISTS(
                      SELECT 1 FROM UNNEST(new_roles) as role 
                      WHERE ENDS_WITH(role, 'Admin')
                    ) THEN 'high' 
                    ELSE 'medium'
                  END as severity,
                  -- Calculate risk score
                  (ARRAY_LENGTH(new_roles) - COALESCE(ARRAY_LENGTH(old_roles), 0)) * 10 as risk_score
                FROM role_changes
                WHERE ARRAY_LENGTH(new_roles) > COALESCE(ARRAY_LENGTH(old_roles), 0)
                ORDER BY timestamp DESC
                """,
                parameters=["project_id", "analysis_hours"]
            ),
            
            "failed_authentications": QueryTemplate(
                template_id="failed_authentications",
                name="Failed Authentication Analysis",
                description="Analyze authentication failures and patterns",
                category="authentication",
                query_sql="""
                SELECT 
                  user_email,
                  source_ip,
                  user_agent,
                  COUNT(*) as failure_count,
                  MIN(timestamp) as first_failure,
                  MAX(timestamp) as last_failure,
                  COUNT(DISTINCT source_ip) as unique_ips,
                  COUNT(DISTINCT user_agent) as unique_user_agents,
                  -- Determine severity based on patterns
                  CASE 
                    WHEN COUNT(*) > 50 OR COUNT(DISTINCT source_ip) > 10 THEN 'high'
                    WHEN COUNT(*) > 20 OR COUNT(DISTINCT source_ip) > 5 THEN 'medium'
                    ELSE 'low'
                  END as severity
                FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_data_access`
                WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                  AND response.status_code >= 400
                  AND (
                    method_name LIKE '%Login%' 
                    OR method_name LIKE '%Auth%'
                    OR response.status_code IN (401, 403)
                  )
                  AND user_email IS NOT NULL
                GROUP BY user_email, source_ip, user_agent
                HAVING COUNT(*) > 5  -- At least 5 failures
                ORDER BY failure_count DESC, unique_ips DESC
                """,
                parameters=["project_id", "analysis_hours"]
            ),
            
            "unusual_api_access": QueryTemplate(
                template_id="unusual_api_access",
                name="Unusual API Access Patterns", 
                description="Detect unusual API access patterns",
                category="api_security",
                query_sql="""
                WITH api_baselines AS (
                  SELECT 
                    user_email,
                    service_name,
                    method_name,
                    AVG(daily_calls) as avg_daily_calls,
                    STDDEV(daily_calls) as stddev_daily_calls
                  FROM (
                    SELECT 
                      user_email,
                      service_name, 
                      method_name,
                      DATE(timestamp) as date,
                      COUNT(*) as daily_calls
                    FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_activity`
                    WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                      AND timestamp < TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                      AND user_email IS NOT NULL
                    GROUP BY user_email, service_name, method_name, date
                  ) daily_stats
                  GROUP BY user_email, service_name, method_name
                  HAVING COUNT(*) >= 7  -- At least 7 days of data
                ),
                recent_calls AS (
                  SELECT 
                    user_email,
                    service_name,
                    method_name,
                    COUNT(*) as recent_calls,
                    MIN(timestamp) as first_call,
                    MAX(timestamp) as last_call,
                    COUNT(DISTINCT source_ip) as unique_ips
                  FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_activity`
                  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                    AND user_email IS NOT NULL
                  GROUP BY user_email, service_name, method_name
                )
                SELECT 
                  r.user_email,
                  r.service_name,
                  r.method_name,
                  r.recent_calls,
                  r.unique_ips,
                  r.first_call,
                  r.last_call,
                  b.avg_daily_calls,
                  -- Calculate anomaly score
                  CASE 
                    WHEN b.stddev_daily_calls > 0 
                    THEN (r.recent_calls - b.avg_daily_calls) / b.stddev_daily_calls
                    ELSE 0 
                  END as anomaly_score,
                  CASE 
                    WHEN r.recent_calls > b.avg_daily_calls * 5 THEN 'critical'
                    WHEN r.recent_calls > b.avg_daily_calls * 3 THEN 'high'
                    WHEN r.recent_calls > b.avg_daily_calls * 2 THEN 'medium'
                    ELSE 'low'
                  END as severity
                FROM recent_calls r
                JOIN api_baselines b USING (user_email, service_name, method_name)
                WHERE r.recent_calls > b.avg_daily_calls * 1.5
                ORDER BY anomaly_score DESC, recent_calls DESC
                """,
                parameters=["project_id", "analysis_hours"]
            ),
            
            "security_metrics": QueryTemplate(
                template_id="security_metrics",
                name="Security Metrics Dashboard",
                description="Generate key security metrics",
                category="metrics",
                query_sql="""
                WITH metrics AS (
                  -- Authentication metrics
                  SELECT 
                    'failed_logins' as metric_name,
                    COUNT(*) as value,
                    'count' as unit,
                    CURRENT_TIMESTAMP() as timestamp
                  FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_data_access`
                  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                    AND response.status_code IN (401, 403)
                  
                  UNION ALL
                  
                  -- API call volume
                  SELECT 
                    'total_api_calls',
                    COUNT(*),
                    'count',
                    CURRENT_TIMESTAMP()
                  FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_activity`
                  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                  
                  UNION ALL
                  
                  -- Unique users
                  SELECT 
                    'active_users',
                    COUNT(DISTINCT user_email),
                    'count', 
                    CURRENT_TIMESTAMP()
                  FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_activity`
                  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                    AND user_email IS NOT NULL
                  
                  UNION ALL
                  
                  -- Error rate
                  SELECT 
                    'api_error_rate',
                    SAFE_DIVIDE(
                      SUM(CASE WHEN response.status_code >= 400 THEN 1 ELSE 0 END),
                      COUNT(*)
                    ) * 100,
                    'percent',
                    CURRENT_TIMESTAMP()
                  FROM `{project_id}.audit_logs.cloudaudit_googleapis_com_activity`
                  WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {analysis_hours} HOUR)
                )
                SELECT * FROM metrics
                ORDER BY metric_name
                """,
                parameters=["project_id", "analysis_hours"]
            )
        }
        
        return templates
    
    def _load_anomaly_thresholds(self) -> Dict[str, Dict[str, float]]:
        """Load anomaly detection thresholds."""
        return {
            "user_behavior": {
                "actions_multiplier_warning": 1.5,
                "actions_multiplier_critical": 3.0,
                "resources_multiplier_warning": 1.2,
                "resources_multiplier_critical": 2.0,
                "z_score_threshold": 2.0
            },
            "api_access": {
                "calls_multiplier_warning": 2.0,
                "calls_multiplier_critical": 5.0,
                "z_score_threshold": 2.5
            },
            "authentication": {
                "failure_count_warning": 10,
                "failure_count_critical": 50,
                "unique_ip_threshold": 5
            }
        }
    
    def _build_query(self, template: QueryTemplate, request: SecurityAnalyticsRequest) -> str:
        """Build SQL query from template and request."""
        query = template.query_sql
        
        # Standard parameters
        params = {
            "project_id": request.project_id,
            "analysis_hours": request.time_range_hours,
            "baseline_days": max(30, request.time_range_hours // 24 * 7)  # Baseline period
        }
        
        # Add custom filters
        params.update(request.filters)
        
        # Format query with parameters
        formatted_query = query.format(**params)
        
        logger.debug(f"Built query for {template.template_id}: {formatted_query}")
        return formatted_query
    
    async def _execute_query(self, query: str) -> List[Dict[str, Any]]:
        """Execute BigQuery and return results."""
        try:
            query_job = self.bq_client.query(query)
            results = []
            
            for row in query_job:
                # Convert BigQuery Row to dict
                row_dict = {}
                for key, value in row.items():
                    if isinstance(value, datetime):
                        row_dict[key] = value.isoformat()
                    elif hasattr(value, '__iter__') and not isinstance(value, (str, bytes)):
                        row_dict[key] = list(value)
                    else:
                        row_dict[key] = value
                results.append(row_dict)
            
            logger.info(f"Query executed successfully, returned {len(results)} rows")
            return results
            
        except Exception as e:
            logger.error(f"BigQuery execution failed: {e}")
            raise
    
    async def _process_results(
        self, 
        query_type: str, 
        results: List[Dict[str, Any]], 
        request: SecurityAnalyticsRequest
    ) -> Dict[str, Any]:
        """Process query results into structured response."""
        
        if query_type == "anomaly_detection":
            return await self._process_anomaly_results(results)
        elif query_type == "privilege_escalation":
            return await self._process_privilege_escalation_results(results)
        elif query_type == "failed_authentications":
            return await self._process_authentication_results(results)
        elif query_type == "unusual_api_access":
            return await self._process_api_access_results(results)
        elif query_type == "security_metrics":
            return await self._process_metrics_results(results)
        else:
            # Generic processing
            return {
                "events": [self._row_to_security_event(row) for row in results],
                "summary": {"total_events": len(results)},
                "insights": [f"Found {len(results)} security events"],
                "recommendations": ["Review security events for potential threats"]
            }
    
    async def _process_anomaly_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process anomaly detection results."""
        anomalies = []
        
        for row in results:
            severity = SecurityEventSeverity(row.get("severity", "low"))
            anomaly = SecurityAnomaly(
                anomaly_id=f"anom_{hash(row.get('user_email', ''))}{int(datetime.utcnow().timestamp())}",
                anomaly_type=AnomalyType.USER_BEHAVIOR,
                severity=severity,
                description=f"User {row.get('user_email')} showed unusual activity patterns",
                affected_user=row.get("user_email"),
                affected_resource="user_behavior",
                detection_time=datetime.utcnow(),
                baseline_behavior={
                    "avg_daily_actions": row.get("avg_daily_actions", 0),
                    "avg_unique_resources": row.get("avg_unique_resources", 0)
                },
                current_behavior={
                    "recent_actions": row.get("recent_actions", 0),
                    "recent_unique_resources": row.get("recent_unique_resources", 0)
                },
                confidence_score=min(1.0, max(0.1, float(row.get("actions_z_score", 1)) / 3)),
                recommended_actions=[
                    "Review user's recent activities",
                    "Verify legitimacy of accessed resources",
                    "Consider implementing additional monitoring"
                ]
            )
            anomalies.append(anomaly)
        
        # Generate insights
        high_severity = [a for a in anomalies if a.severity == SecurityEventSeverity.HIGH]
        insights = [
            f"Detected {len(anomalies)} behavioral anomalies",
            f"{len(high_severity)} require immediate attention"
        ]
        
        recommendations = []
        if high_severity:
            recommendations.extend([
                "Investigate high-severity anomalies immediately",
                "Implement user behavior analytics for continuous monitoring"
            ])
        
        return {
            "anomalies": anomalies,
            "summary": {
                "total_anomalies": len(anomalies),
                "high_severity": len(high_severity),
                "affected_users": len(set(a.affected_user for a in anomalies))
            },
            "insights": insights,
            "recommendations": recommendations
        }
    
    async def _process_privilege_escalation_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process privilege escalation results."""
        events = []
        
        for row in results:
            severity = SecurityEventSeverity(row.get("severity", "medium"))
            event = SecurityEvent(
                timestamp=datetime.fromisoformat(row["timestamp"]),
                event_type="privilege_escalation",
                severity=severity,
                user_email=row.get("user_email"),
                resource=row.get("resource_name", "unknown"),
                action="iam_policy_change",
                details={
                    "new_roles": row.get("new_roles", []),
                    "old_roles": row.get("old_roles", []),
                    "risk_score": row.get("risk_score", 0)
                },
                risk_score=min(100, max(0, int(row.get("risk_score", 0))))
            )
            events.append(event)
        
        critical_events = [e for e in events if e.severity == SecurityEventSeverity.CRITICAL]
        
        return {
            "events": events,
            "summary": {
                "total_escalations": len(events),
                "critical_escalations": len(critical_events)
            },
            "insights": [
                f"Detected {len(events)} privilege escalation events",
                f"{len(critical_events)} involve critical roles"
            ],
            "recommendations": [
                "Review all privilege escalation events",
                "Implement approval workflow for sensitive role assignments",
                "Enable real-time alerting for critical role changes"
            ]
        }
    
    async def _process_authentication_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process authentication failure results."""
        events = []
        
        for row in results:
            severity = SecurityEventSeverity(row.get("severity", "low"))
            event = SecurityEvent(
                timestamp=datetime.fromisoformat(row["first_failure"]),
                event_type="authentication_failure",
                severity=severity,
                user_email=row.get("user_email"),
                resource="authentication_service",
                action="failed_login",
                source_ip=row.get("source_ip"),
                user_agent=row.get("user_agent"),
                details={
                    "failure_count": row.get("failure_count", 0),
                    "time_span": {
                        "first": row.get("first_failure"),
                        "last": row.get("last_failure")
                    },
                    "unique_ips": row.get("unique_ips", 0),
                    "unique_user_agents": row.get("unique_user_agents", 0)
                },
                risk_score=min(100, int(row.get("failure_count", 0) * 2))
            )
            events.append(event)
        
        return {
            "events": events,
            "summary": {
                "failed_auth_patterns": len(events),
                "total_failures": sum(e.details["failure_count"] for e in events)
            },
            "insights": [
                f"Identified {len(events)} suspicious authentication patterns",
                "Multiple failed attempts may indicate brute force attacks"
            ],
            "recommendations": [
                "Implement account lockout policies",
                "Enable MFA for all accounts",
                "Monitor and block suspicious IP addresses"
            ]
        }
    
    async def _process_api_access_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process unusual API access results.""" 
        anomalies = []
        
        for row in results:
            severity = SecurityEventSeverity(row.get("severity", "low"))
            anomaly = SecurityAnomaly(
                anomaly_id=f"api_{hash(f'{row.get(\"user_email\")}_{row.get(\"service_name\")}')}{int(datetime.utcnow().timestamp())}",
                anomaly_type=AnomalyType.API_ACCESS,
                severity=severity,
                description=f"Unusual API access pattern: {row.get('method_name')} on {row.get('service_name')}",
                affected_user=row.get("user_email"),
                affected_resource=f"{row.get('service_name')}/{row.get('method_name')}",
                detection_time=datetime.utcnow(),
                baseline_behavior={
                    "avg_daily_calls": row.get("avg_daily_calls", 0)
                },
                current_behavior={
                    "recent_calls": row.get("recent_calls", 0),
                    "unique_ips": row.get("unique_ips", 0)
                },
                confidence_score=min(1.0, abs(row.get("anomaly_score", 1)) / 5),
                recommended_actions=[
                    "Review API usage patterns",
                    "Verify business justification for increased usage",
                    "Check for potential data exfiltration"
                ]
            )
            anomalies.append(anomaly)
        
        return {
            "anomalies": anomalies,
            "summary": {
                "unusual_patterns": len(anomalies),
                "services_affected": len(set(a.affected_resource.split('/')[0] for a in anomalies))
            },
            "insights": [
                f"Detected {len(anomalies)} unusual API access patterns",
                "Monitor for potential data exfiltration or abuse"
            ],
            "recommendations": [
                "Implement API rate limiting",
                "Set up alerts for unusual API usage",
                "Regular review of API access patterns"
            ]
        }
    
    async def _process_metrics_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process security metrics results."""
        metrics = []
        
        for row in results:
            metric = SecurityMetrics(
                metric_name=row["metric_name"],
                value=float(row["value"]),
                unit=row["unit"],
                timestamp=datetime.fromisoformat(row["timestamp"]),
                tags={"source": "bigquery_analytics"}
            )
            metrics.append(metric)
        
        # Calculate summary statistics
        metrics_dict = {m.metric_name: m.value for m in metrics}
        
        return {
            "metrics": metrics,
            "summary": {
                "total_metrics": len(metrics),
                "key_values": metrics_dict
            },
            "insights": [
                f"Generated {len(metrics)} security metrics",
                "Use these metrics for security dashboard and alerting"
            ],
            "recommendations": [
                "Set up automated alerting based on metric thresholds",
                "Create security dashboards for continuous monitoring"
            ]
        }
    
    def _row_to_security_event(self, row: Dict[str, Any]) -> SecurityEvent:
        """Convert BigQuery row to SecurityEvent."""
        return SecurityEvent(
            timestamp=datetime.fromisoformat(row.get("timestamp", datetime.utcnow().isoformat())),
            event_type=row.get("event_type", "unknown"),
            severity=SecurityEventSeverity(row.get("severity", "low")),
            user_email=row.get("user_email"),
            resource=row.get("resource", "unknown"),
            action=row.get("action", "unknown"),
            source_ip=row.get("source_ip"),
            user_agent=row.get("user_agent"),
            details=row,
            risk_score=row.get("risk_score", 0)
        )
    
    async def get_security_dashboard(self, project_id: str) -> SecurityDashboard:
        """Generate comprehensive security dashboard."""
        # Run multiple analytics in parallel
        dashboard_queries = [
            SecurityAnalyticsRequest(
                query_type="security_metrics",
                project_id=project_id,
                time_range_hours=24
            ),
            SecurityAnalyticsRequest(
                query_type="anomaly_detection", 
                project_id=project_id,
                time_range_hours=24
            )
        ]
        
        results = await asyncio.gather(*[
            self.run_security_analytics(req) for req in dashboard_queries
        ])
        
        metrics_result, anomalies_result = results
        
        # Calculate security score (simplified)
        security_score = 100
        if anomalies_result.success and anomalies_result.anomalies:
            high_anomalies = [a for a in anomalies_result.anomalies if a.severity == SecurityEventSeverity.HIGH]
            security_score = max(0, 100 - len(high_anomalies) * 10 - len(anomalies_result.anomalies) * 2)
        
        return SecurityDashboard(
            dashboard_id=f"dash_{int(datetime.utcnow().timestamp())}",
            project_id=project_id,
            generated_at=datetime.utcnow(),
            security_score=security_score,
            threat_level=SecurityEventSeverity.LOW,  # Would calculate based on recent events
            active_incidents=0,  # Would query incident system
            resolved_incidents_24h=0,
            recent_anomalies=anomalies_result.anomalies[:5] if anomalies_result.success else [],
            compliance_score=85,  # Would calculate from compliance analysis
            compliance_violations=[],
            top_recommendations=[
                "Enable continuous security monitoring",
                "Implement automated threat response",
                "Regular security assessments"
            ]
        )
    
    async def _generate_sample_results(self, request: SecurityAnalyticsRequest) -> Dict[str, Any]:
        """Generate sample results when BigQuery is not available."""
        logger.info(f"Generating sample results for {request.query_type}")
        
        if request.query_type == "anomaly_detection":
            return {
                "anomalies": [
                    SecurityAnomaly(
                        anomaly_id="sample_anom_1",
                        anomaly_type=AnomalyType.USER_BEHAVIOR,
                        severity=SecurityEventSeverity.MEDIUM,
                        description="Sample anomaly: Unusual activity detected",
                        affected_user="sample.user@example.com",
                        affected_resource="sample_resource",
                        detection_time=datetime.utcnow(),
                        baseline_behavior={"avg_daily_actions": 50},
                        current_behavior={"recent_actions": 150},
                        confidence_score=0.75,
                        recommended_actions=["Review user activity", "Monitor for 24 hours"]
                    )
                ],
                "summary": {
                    "total_anomalies": 1,
                    "high_severity": 0,
                    "affected_users": 1
                },
                "insights": ["Sample data - BigQuery not configured"],
                "recommendations": ["Configure BigQuery for real analytics"]
            }
        
        elif request.query_type == "security_metrics":
            return {
                "metrics": [
                    SecurityMetrics(
                        metric_name="failed_logins",
                        value=5.0,
                        unit="count",
                        timestamp=datetime.utcnow(),
                        tags={"source": "sample_data"}
                    ),
                    SecurityMetrics(
                        metric_name="active_users",
                        value=100.0,
                        unit="count",
                        timestamp=datetime.utcnow(),
                        tags={"source": "sample_data"}
                    )
                ],
                "summary": {
                    "total_metrics": 2,
                    "key_values": {"failed_logins": 5.0, "active_users": 100.0}
                },
                "insights": ["Sample metrics data"],
                "recommendations": ["Enable BigQuery for real-time metrics"]
            }
        
        else:
            # Generic sample response
            return {
                "events": [
                    SecurityEvent(
                        timestamp=datetime.utcnow(),
                        event_type="sample_event",
                        severity=SecurityEventSeverity.LOW,
                        user_email="sample@example.com",
                        resource="sample_resource",
                        action="sample_action",
                        details={"note": "Sample data - configure BigQuery for real data"},
                        risk_score=10
                    )
                ],
                "summary": {"total_events": 1},
                "insights": ["Sample data provided"],
                "recommendations": ["Configure BigQuery analytics"]
            }