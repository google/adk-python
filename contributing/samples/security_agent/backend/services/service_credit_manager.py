"""
Service Credit Template Manager Service
======================================

Service for generating Google Cloud service credit request templates,
managing incident claims, analyzing SLA violations, and tracking credit approvals.
"""

import os
import logging
import sqlite3
import json
import asyncio
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

try:
    from google.cloud import monitoring_v3
    from google.cloud import logging_v1
    from google.cloud import support_v2
    from google.cloud import billing_v1
    GCLOUD_AVAILABLE = True
except ImportError:
    GCLOUD_AVAILABLE = False

from ..models.service_credit_models import (
    IncidentSeverity, ServiceType, SLAViolationType, CreditRequestStatus,
    EvidenceType, ImpactScope, BusinessImpact, SLAMetrics, IncidentEvidence,
    ServiceIncident, CreditCalculation, ServiceCreditTemplate, ServiceCreditRequest,
    CreditRequestFilters, CreditAnalytics, TemplateGenerationRequest,
    CreditRequestResponse
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ServiceCreditManager:
    """Service Credit Template Manager"""
    
    def __init__(self, project_id: str, organization_id: Optional[str] = None,
                 database_path: str = "backend/cache/gcp_data.db"):
        self.project_id = project_id
        self.organization_id = organization_id
        self.database_path = database_path
        
        # Initialize GCP clients if available
        if GCLOUD_AVAILABLE:
            try:
                self.monitoring_client = monitoring_v3.MetricServiceClient()
                self.logging_client = logging_v1.Client(project=project_id)
                self.billing_client = billing_v1.CloudBillingClient()
                logger.info("GCP clients initialized successfully")
            except Exception as e:
                logger.warning(f"Failed to initialize some GCP clients: {e}")
                self.monitoring_client = None
                self.logging_client = None
                self.billing_client = None
        else:
            self.monitoring_client = None
            self.logging_client = None
            self.billing_client = None
        
        # Initialize database
        self._init_database()
        
        # Load SLA thresholds and service configurations
        self.sla_thresholds = self._load_sla_thresholds()
        self.service_configs = self._load_service_configurations()
        
        # Template library
        self.template_library = self._load_template_library()
    
    def _init_database(self):
        """Initialize database tables for service credit management"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Service credit templates table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS service_credit_templates (
                    template_id TEXT PRIMARY KEY,
                    template_name TEXT,
                    service_type TEXT,
                    violation_type TEXT,
                    created_at TIMESTAMP,
                    created_by TEXT,
                    description TEXT,
                    incident_details_template TEXT,
                    business_impact_template TEXT,
                    technical_details_template TEXT,
                    evidence_requirements JSON,
                    sla_reference TEXT,
                    credit_calculation_formula TEXT,
                    usage_count INTEGER DEFAULT 0,
                    success_rate REAL DEFAULT 0.0,
                    average_processing_days INTEGER,
                    tags JSON,
                    is_active INTEGER DEFAULT 1
                )
            """)
            
            # Service credit requests table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS service_credit_requests (
                    request_id TEXT PRIMARY KEY,
                    template_id TEXT,
                    created_at TIMESTAMP,
                    created_by TEXT,
                    status TEXT,
                    project_id TEXT,
                    billing_account TEXT,
                    organization_id TEXT,
                    incident_data JSON,
                    credit_calculation JSON,
                    justification TEXT,
                    additional_context TEXT,
                    submitted_at TIMESTAMP,
                    reviewed_at TIMESTAMP,
                    reviewer TEXT,
                    review_notes TEXT,
                    approved_amount REAL,
                    rejection_reason TEXT,
                    follow_up_required INTEGER DEFAULT 0,
                    follow_up_date TIMESTAMP,
                    escalation_level INTEGER DEFAULT 0,
                    FOREIGN KEY (template_id) REFERENCES service_credit_templates(template_id)
                )
            """)
            
            # Incident evidence table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS incident_evidence (
                    evidence_id TEXT PRIMARY KEY,
                    request_id TEXT,
                    evidence_type TEXT,
                    title TEXT,
                    description TEXT,
                    file_path TEXT,
                    url TEXT,
                    timestamp TIMESTAMP,
                    relevance_score REAL,
                    source_system TEXT,
                    metadata JSON,
                    FOREIGN KEY (request_id) REFERENCES service_credit_requests(request_id)
                )
            """)
            
            # SLA violations tracking table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sla_violations (
                    violation_id TEXT PRIMARY KEY,
                    service_type TEXT,
                    violation_type TEXT,
                    detected_at TIMESTAMP,
                    resolved_at TIMESTAMP,
                    duration_minutes INTEGER,
                    affected_resources JSON,
                    impact_scope TEXT,
                    severity TEXT,
                    sla_metrics JSON,
                    credit_eligible INTEGER DEFAULT 0,
                    credit_requested INTEGER DEFAULT 0,
                    credit_approved INTEGER DEFAULT 0,
                    notes TEXT
                )
            """)
            
            conn.commit()
            conn.close()
            logger.info("Service credit database initialized")
            
        except Exception as e:
            logger.error(f"Failed to initialize service credit database: {e}")
    
    def _load_sla_thresholds(self) -> Dict[str, Dict[str, Any]]:
        """Load SLA thresholds for different services"""
        return {
            "COMPUTE_ENGINE": {
                "monthly_uptime": 99.95,  # %
                "instance_availability": 99.5,  # %
                "network_uptime": 99.99  # %
            },
            "APP_ENGINE": {
                "standard_uptime": 99.95,  # %
                "flexible_uptime": 99.5  # %
            },
            "KUBERNETES_ENGINE": {
                "control_plane_uptime": 99.95,  # %
                "regional_cluster_uptime": 99.99,  # %
                "zonal_cluster_uptime": 99.5  # %
            },
            "CLOUD_STORAGE": {
                "standard_availability": 99.95,  # %
                "nearline_availability": 99.9,  # %
                "coldline_availability": 99.9,  # %
                "archive_availability": 99.9,  # %
                "durability": 99.999999999  # 11 9's
            },
            "CLOUD_SQL": {
                "standard_uptime": 99.95,  # %
                "high_availability_uptime": 99.99  # %
            },
            "BIG_QUERY": {
                "monthly_uptime": 99.99,  # %
                "query_availability": 99.9  # %
            }
        }
    
    def _load_service_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Load service-specific configurations"""
        return {
            "COMPUTE_ENGINE": {
                "credit_percentage": 10,  # % of monthly bill
                "max_credit_percentage": 100,
                "minimum_outage_minutes": 5,
                "evidence_requirements": ["MONITORING_DATA", "ERROR_LOGS"],
                "sla_document": "https://cloud.google.com/compute/sla"
            },
            "KUBERNETES_ENGINE": {
                "credit_percentage": 10,
                "max_credit_percentage": 100,
                "minimum_outage_minutes": 5,
                "evidence_requirements": ["MONITORING_DATA", "ERROR_LOGS", "INCIDENT_REPORT"],
                "sla_document": "https://cloud.google.com/kubernetes-engine/sla"
            },
            "CLOUD_STORAGE": {
                "credit_percentage": 25,
                "max_credit_percentage": 100,
                "minimum_outage_minutes": 1,
                "evidence_requirements": ["ERROR_LOGS", "SUPPORT_TICKET"],
                "sla_document": "https://cloud.google.com/storage/sla"
            }
        }
    
    def _load_template_library(self) -> Dict[str, str]:
        """Load pre-built template library"""
        return {
            "COMPUTE_ENGINE_AVAILABILITY": """
# Compute Engine Service Credit Request

## Incident Summary
On {incident_date}, our Compute Engine instances experienced {severity} availability issues affecting {affected_regions} for approximately {duration_minutes} minutes.

## Business Impact
- **Affected Users**: {affected_users}
- **Revenue Impact**: ${revenue_impact}
- **Service Degradation**: {degradation_percentage}%
- **Critical Functions Affected**: {critical_functions}

## Technical Details
- **Instance Types**: {instance_types}
- **Zones Affected**: {affected_zones}
- **Error Messages**: {error_messages}
- **Root Cause**: {root_cause}

## SLA Violation
Our monthly uptime fell below the guaranteed 99.95% SLA to {actual_uptime}%, resulting in a breach of {breach_percentage}%.

## Supporting Evidence
1. Cloud Monitoring dashboards showing instance unavailability
2. Error logs from affected instances
3. Support ticket #{support_ticket_id}
4. Customer impact notifications

## Credit Calculation
Based on the Compute Engine SLA, we are eligible for {credit_percentage}% of our monthly Compute Engine charges.
- Monthly Charges: ${monthly_charges}
- Credit Amount: ${credit_amount}
""",
            
            "CLOUD_STORAGE_AVAILABILITY": """
# Cloud Storage Service Credit Request

## Incident Summary
Cloud Storage bucket access was impacted on {incident_date} in {affected_regions} for {duration_minutes} minutes, violating the {storage_class} SLA.

## Business Impact
- **Applications Affected**: {affected_applications}
- **Data Access Failures**: {access_failures}
- **Customer Impact**: {customer_impact}
- **Backup Operations**: {backup_impact}

## Technical Details
- **Affected Buckets**: {affected_buckets}
- **Storage Classes**: {storage_classes}
- **Error Rates**: {error_rates}
- **API Failures**: {api_failures}

## SLA Violation
Cloud Storage availability fell below {sla_threshold}% to {actual_availability}%.

## Evidence
1. Storage access logs showing failures
2. Application error logs
3. Monitoring alerts
4. Support case documentation

## Credit Request
Requesting {credit_percentage}% credit of monthly storage charges: ${credit_amount}
"""
        }
    
    async def generate_template(self, request: TemplateGenerationRequest) -> ServiceCreditTemplate:
        """Generate a service credit template"""
        logger.info(f"Generating template for {request.service_type} - {request.violation_type}")
        
        try:
            # Get service configuration
            service_config = self.service_configs.get(request.service_type.value, {})
            
            # Generate template name if not provided
            template_name = request.template_name or f"{request.service_type.value}_{request.violation_type.value}_Template"
            
            # Build evidence requirements
            evidence_requirements = service_config.get('evidence_requirements', [])
            evidence_requirements.extend([
                EvidenceType.MONITORING_DATA,
                EvidenceType.SUPPORT_TICKET,
                EvidenceType.INCIDENT_REPORT
            ])
            
            # Generate template sections
            incident_template = self._generate_incident_template(request.service_type, request.violation_type)
            impact_template = self._generate_business_impact_template()
            technical_template = self._generate_technical_details_template(request.service_type)
            
            # Create template
            template = ServiceCreditTemplate(
                template_name=template_name,
                service_type=request.service_type,
                violation_type=request.violation_type,
                description=f"Template for {request.service_type.value} {request.violation_type.value} incidents",
                incident_details_template=incident_template,
                business_impact_template=impact_template,
                technical_details_template=technical_template,
                evidence_requirements=[EvidenceType(er) for er in evidence_requirements],
                sla_reference=service_config.get('sla_document', 'https://cloud.google.com/terms/sla'),
                credit_calculation_formula=self._generate_credit_formula(request.service_type),
                tags=[request.service_type.value, request.violation_type.value, "auto-generated"]
            )
            
            # Save template to database
            await self._save_template(template)
            
            logger.info(f"Template generated: {template.template_id}")
            return template
            
        except Exception as e:
            logger.error(f"Template generation failed: {e}")
            raise
    
    def _generate_incident_template(self, service_type: ServiceType, violation_type: SLAViolationType) -> str:
        """Generate incident details template"""
        base_template = """
## Incident Overview
- **Service**: {service_name}
- **Incident Date**: {incident_date}
- **Start Time**: {start_time}
- **End Time**: {end_time}
- **Duration**: {duration_minutes} minutes
- **Severity**: {severity}
- **Regions Affected**: {affected_regions}

## Incident Description
{incident_description}

## Root Cause
{root_cause}

## Google Incident Reference
{google_incident_id}
"""
        
        # Customize based on service type
        if service_type == ServiceType.COMPUTE_ENGINE:
            base_template += """
## Affected Resources
- **Instance Types**: {instance_types}
- **Zones**: {affected_zones}
- **Instance Count**: {instance_count}
"""
        elif service_type == ServiceType.CLOUD_STORAGE:
            base_template += """
## Affected Resources
- **Buckets**: {affected_buckets}
- **Storage Classes**: {storage_classes}
- **Regions**: {storage_regions}
"""
        
        return base_template
    
    def _generate_business_impact_template(self) -> str:
        """Generate business impact template"""
        return """
## Business Impact Assessment

### User Impact
- **Users Affected**: {affected_users}
- **Customer Complaints**: {customer_complaints}
- **Service Availability**: {availability_percentage}%

### Financial Impact
- **Estimated Revenue Loss**: ${revenue_impact}
- **Mitigation Costs**: ${mitigation_costs}
- **SLA Credits Due**: ${sla_credits}

### Operational Impact
- **Critical Functions Affected**: {critical_functions}
- **Backup/DR Activated**: {dr_activated}
- **Staff Hours**: {staff_hours}

### Reputation Impact
{reputation_impact}

### Customer Communications
{customer_communications}
"""
    
    def _generate_technical_details_template(self, service_type: ServiceType) -> str:
        """Generate technical details template"""
        base_template = """
## Technical Analysis

### Error Messages
```
{error_messages}
```

### Affected Components
{affected_components}

### Monitoring Data
- **CPU Usage**: {cpu_usage}%
- **Memory Usage**: {memory_usage}%
- **Network I/O**: {network_io}
- **Error Rate**: {error_rate}%

### Logs Analysis
{logs_analysis}
"""
        
        # Add service-specific sections
        if service_type == ServiceType.KUBERNETES_ENGINE:
            base_template += """
### Kubernetes Specific
- **Cluster Version**: {cluster_version}
- **Node Pool Status**: {node_pool_status}
- **Pod Failures**: {pod_failures}
- **Service Mesh Impact**: {service_mesh_impact}
"""
        
        return base_template
    
    def _generate_credit_formula(self, service_type: ServiceType) -> str:
        """Generate credit calculation formula"""
        config = self.service_configs.get(service_type.value, {})
        credit_percentage = config.get('credit_percentage', 10)
        
        return f"""
Credit Amount = (Monthly Service Charges × {credit_percentage}%) × (Downtime Minutes / Total Minutes in Month)

Where:
- Monthly Service Charges: Total charges for {service_type.value} in the affected billing period
- Downtime Minutes: Total minutes of service unavailability
- Maximum Credit: {config.get('max_credit_percentage', 100)}% of monthly charges
"""
    
    async def create_credit_request(self, incident: ServiceIncident, 
                                  template_id: Optional[str] = None) -> CreditRequestResponse:
        """Create a service credit request from incident data"""
        logger.info(f"Creating credit request for incident: {incident.incident_id}")
        
        try:
            # Calculate credit amount
            credit_calc = await self._calculate_service_credit(incident)
            
            # Generate justification
            justification = await self._generate_justification(incident, credit_calc)
            
            # Create credit request
            credit_request = ServiceCreditRequest(
                template_id=template_id,
                created_by="system",
                project_id=self.project_id,
                billing_account=self._get_billing_account(),
                organization_id=self.organization_id,
                incident=incident,
                credit_calculation=credit_calc,
                justification=justification
            )
            
            # Validate request
            validation_errors = await self._validate_credit_request(credit_request)
            
            # Generate recommendations
            recommendations = await self._generate_recommendations(credit_request)
            
            # Calculate approval probability
            approval_probability = await self._estimate_approval_probability(credit_request)
            
            # Find similar cases
            similar_cases = await self._find_similar_cases(credit_request)
            
            # Save request to database
            if not validation_errors:
                await self._save_credit_request(credit_request)
            
            response = CreditRequestResponse(
                success=len(validation_errors) == 0,
                request=credit_request if not validation_errors else None,
                validation_errors=validation_errors,
                recommendations=recommendations,
                estimated_approval_probability=approval_probability,
                similar_cases=similar_cases,
                next_steps=self._generate_next_steps(credit_request, validation_errors),
                processing_time_estimate=self._estimate_processing_time(credit_request)
            )
            
            logger.info(f"Credit request created: {credit_request.request_id}")
            return response
            
        except Exception as e:
            logger.error(f"Failed to create credit request: {e}")
            return CreditRequestResponse(
                success=False,
                validation_errors=[f"Failed to create request: {str(e)}"]
            )
    
    async def _calculate_service_credit(self, incident: ServiceIncident) -> CreditCalculation:
        """Calculate service credit amount"""
        try:
            # Get billing data for the affected period
            monthly_charges = await self._get_monthly_charges(
                incident.service_type, 
                incident.start_time
            )
            
            # Get service configuration
            config = self.service_configs.get(incident.service_type.value, {})
            credit_percentage = config.get('credit_percentage', 10)
            max_credit_percentage = config.get('max_credit_percentage', 100)
            
            # Calculate affected percentage based on impact scope
            affected_percentage = self._calculate_affected_percentage(incident)
            
            # Calculate base credit
            sla_credit_percentage = credit_percentage
            if incident.severity == IncidentSeverity.CRITICAL:
                sla_credit_percentage = min(credit_percentage * 2, max_credit_percentage)
            
            calculated_credit = (monthly_charges * affected_percentage / 100) * (sla_credit_percentage / 100)
            maximum_credit = monthly_charges * (max_credit_percentage / 100)
            
            return CreditCalculation(
                base_charges=monthly_charges,
                affected_percentage=affected_percentage,
                sla_credit_percentage=sla_credit_percentage,
                calculated_credit=calculated_credit,
                maximum_credit=maximum_credit,
                final_credit_amount=min(calculated_credit, maximum_credit),
                calculation_method="Standard SLA credit calculation",
                billing_period=incident.start_time.strftime("%Y-%m")
            )
            
        except Exception as e:
            logger.error(f"Credit calculation failed: {e}")
            # Return minimal calculation
            return CreditCalculation(
                base_charges=1000.0,  # Default estimate
                affected_percentage=50.0,
                sla_credit_percentage=10.0,
                calculated_credit=50.0,
                maximum_credit=1000.0,
                final_credit_amount=50.0,
                calculation_method="Estimated calculation (billing data unavailable)",
                billing_period=incident.start_time.strftime("%Y-%m")
            )
    
    async def _get_monthly_charges(self, service_type: ServiceType, date: datetime) -> float:
        """Get monthly charges for a service"""
        try:
            if self.billing_client:
                # In production, this would query the Cloud Billing API
                # For now, return mock data
                pass
            
            # Mock billing data based on service type
            mock_charges = {
                ServiceType.COMPUTE_ENGINE: 2500.0,
                ServiceType.KUBERNETES_ENGINE: 1800.0,
                ServiceType.CLOUD_STORAGE: 500.0,
                ServiceType.CLOUD_SQL: 1200.0,
                ServiceType.BIG_QUERY: 800.0
            }
            
            return mock_charges.get(service_type, 1000.0)
            
        except Exception as e:
            logger.error(f"Failed to get billing data: {e}")
            return 1000.0  # Default estimate
    
    def _calculate_affected_percentage(self, incident: ServiceIncident) -> float:
        """Calculate percentage of service affected"""
        # Base calculation on impact scope
        scope_percentages = {
            ImpactScope.GLOBAL: 100.0,
            ImpactScope.REGIONAL: 50.0,
            ImpactScope.ZONAL: 25.0,
            ImpactScope.PROJECT_WIDE: 75.0,
            ImpactScope.RESOURCE_SPECIFIC: 10.0
        }
        
        base_percentage = scope_percentages.get(incident.impact_scope, 50.0)
        
        # Adjust based on severity
        severity_multipliers = {
            IncidentSeverity.CRITICAL: 1.0,
            IncidentSeverity.HIGH: 0.8,
            IncidentSeverity.MEDIUM: 0.6,
            IncidentSeverity.LOW: 0.4
        }
        
        multiplier = severity_multipliers.get(incident.severity, 1.0)
        
        return min(base_percentage * multiplier, 100.0)
    
    def _get_billing_account(self) -> str:
        """Get billing account for the project"""
        # In production, this would query the project's billing account
        return os.getenv('BILLING_ACCOUNT_ID', 'mock-billing-account')
    
    async def _generate_justification(self, incident: ServiceIncident, 
                                    credit_calc: CreditCalculation) -> str:
        """Generate justification text for credit request"""
        justification = f"""
Service Credit Request Justification

On {incident.start_time.strftime('%Y-%m-%d %H:%M UTC')}, our {incident.service_type.value} service experienced a {incident.severity.value} incident that lasted {incident.duration_minutes} minutes.

SLA Violation:
This incident violated our Service Level Agreement, which guarantees {self._get_sla_threshold(incident.service_type)}% availability. The actual availability during this period fell below the committed threshold.

Business Impact:
- {incident.business_impact.affected_users or 'Multiple'} users were affected
- Estimated revenue impact: ${incident.business_impact.revenue_impact or 'TBD'}
- Service degradation: {incident.business_impact.service_degradation_percentage or 'Significant'}%

Credit Calculation:
Based on our monthly {incident.service_type.value} charges of ${credit_calc.base_charges:,.2f} and the {credit_calc.affected_percentage:.1f}% impact, we are requesting a service credit of ${credit_calc.final_credit_amount:,.2f}.

This credit request is submitted in accordance with the Google Cloud Platform Service Level Agreement and our commitment to service reliability.
"""
        
        return justification
    
    def _get_sla_threshold(self, service_type: ServiceType) -> float:
        """Get SLA threshold for service type"""
        thresholds = self.sla_thresholds.get(service_type.value, {})
        return thresholds.get('monthly_uptime', 99.9)
    
    async def _validate_credit_request(self, request: ServiceCreditRequest) -> List[str]:
        """Validate credit request"""
        errors = []
        
        # Check incident duration
        if request.incident.duration_minutes and request.incident.duration_minutes < 5:
            errors.append("Incident duration must be at least 5 minutes for credit eligibility")
        
        # Check if evidence is provided
        if not request.incident.evidence:
            errors.append("At least one piece of evidence is required")
        
        # Check credit amount
        if request.credit_calculation.final_credit_amount <= 0:
            errors.append("Credit amount must be greater than zero")
        
        # Check for required fields
        if not request.incident.root_cause:
            errors.append("Root cause analysis is required")
        
        if not request.billing_account:
            errors.append("Billing account must be specified")
        
        return errors
    
    async def _generate_recommendations(self, request: ServiceCreditRequest) -> List[str]:
        """Generate recommendations to improve credit request"""
        recommendations = []
        
        # Check evidence strength
        evidence_types = [e.evidence_type for e in request.incident.evidence]
        
        if EvidenceType.MONITORING_DATA not in evidence_types:
            recommendations.append("Add monitoring data to strengthen your claim")
        
        if EvidenceType.SUPPORT_TICKET not in evidence_types:
            recommendations.append("Reference existing support ticket if available")
        
        if not request.incident.google_incident_id:
            recommendations.append("Include Google's incident ID if this was a known issue")
        
        # Check business impact documentation
        if not request.incident.business_impact.revenue_impact:
            recommendations.append("Quantify revenue impact if possible")
        
        if not request.incident.customer_communications:
            recommendations.append("Document customer communications during incident")
        
        return recommendations
    
    async def _estimate_approval_probability(self, request: ServiceCreditRequest) -> float:
        """Estimate probability of credit request approval"""
        score = 0.5  # Base probability
        
        # Increase score for strong evidence
        if len(request.incident.evidence) >= 3:
            score += 0.2
        
        # Increase for clear SLA violation
        if request.incident.sla_violations:
            score += 0.2
        
        # Increase for detailed impact assessment
        if request.incident.business_impact.affected_users:
            score += 0.1
        
        # Decrease for very short incidents
        if request.incident.duration_minutes and request.incident.duration_minutes < 15:
            score -= 0.1
        
        return min(max(score, 0.0), 1.0)
    
    async def _find_similar_cases(self, request: ServiceCreditRequest) -> List[Dict[str, Any]]:
        """Find similar historical cases"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                SELECT request_id, status, approved_amount, credit_calculation
                FROM service_credit_requests
                WHERE incident_data LIKE ?
                  AND status IN ('APPROVED', 'PARTIALLY_APPROVED')
                ORDER BY created_at DESC
                LIMIT 5
            """, (f'%{request.incident.service_type.value}%',))
            
            rows = cursor.fetchall()
            conn.close()
            
            similar_cases = []
            for row in rows:
                similar_cases.append({
                    'request_id': row[0],
                    'status': row[1],
                    'approved_amount': row[2],
                    'similarity_score': 0.8  # Mock similarity
                })
            
            return similar_cases
            
        except Exception as e:
            logger.error(f"Failed to find similar cases: {e}")
            return []
    
    def _generate_next_steps(self, request: ServiceCreditRequest, 
                           validation_errors: List[str]) -> List[str]:
        """Generate next steps for the user"""
        if validation_errors:
            return [
                "Fix validation errors before submitting",
                "Gather additional evidence as recommended",
                "Review and refine business impact assessment"
            ]
        
        return [
            "Review the generated credit request",
            "Submit the request through Google Cloud Console",
            "Monitor request status and respond to any follow-up questions",
            "Prepare for potential clarification requests from Google Support"
        ]
    
    def _estimate_processing_time(self, request: ServiceCreditRequest) -> int:
        """Estimate processing time in days"""
        base_days = 7  # Base processing time
        
        # Add time for complex cases
        if request.credit_calculation.final_credit_amount > 10000:
            base_days += 3
        
        if request.incident.severity == IncidentSeverity.CRITICAL:
            base_days += 2
        
        if not request.incident.google_incident_id:
            base_days += 2  # Additional investigation needed
        
        return base_days
    
    async def _save_template(self, template: ServiceCreditTemplate):
        """Save template to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO service_credit_templates
                (template_id, template_name, service_type, violation_type, created_at,
                 description, incident_details_template, business_impact_template,
                 technical_details_template, evidence_requirements, sla_reference,
                 credit_calculation_formula, tags)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                template.template_id,
                template.template_name,
                template.service_type.value,
                template.violation_type.value,
                template.created_at,
                template.description,
                template.incident_details_template,
                template.business_impact_template,
                template.technical_details_template,
                json.dumps([er.value for er in template.evidence_requirements]),
                template.sla_reference,
                template.credit_calculation_formula,
                json.dumps(template.tags)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save template: {e}")
    
    async def _save_credit_request(self, request: ServiceCreditRequest):
        """Save credit request to database"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            cursor.execute("""
                INSERT INTO service_credit_requests
                (request_id, template_id, created_at, created_by, status,
                 project_id, billing_account, organization_id, incident_data,
                 credit_calculation, justification, additional_context)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                request.request_id,
                request.template_id,
                request.created_at,
                request.created_by,
                request.status.value,
                request.project_id,
                request.billing_account,
                request.organization_id,
                json.dumps(request.incident.dict(), default=str),
                json.dumps(request.credit_calculation.dict()),
                request.justification,
                request.additional_context
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            logger.error(f"Failed to save credit request: {e}")
    
    async def get_templates(self, service_type: Optional[ServiceType] = None) -> List[ServiceCreditTemplate]:
        """Get available templates"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM service_credit_templates WHERE is_active = 1"
            params = []
            
            if service_type:
                query += " AND service_type = ?"
                params.append(service_type.value)
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            templates = []
            # Convert rows to ServiceCreditTemplate objects
            # (Implementation would properly deserialize)
            
            return templates
            
        except Exception as e:
            logger.error(f"Failed to get templates: {e}")
            return []
    
    async def get_credit_requests(self, filters: Optional[CreditRequestFilters] = None) -> List[ServiceCreditRequest]:
        """Get credit requests with optional filters"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM service_credit_requests WHERE 1=1"
            params = []
            
            if filters:
                if filters.status:
                    status_placeholders = ','.join('?' for _ in filters.status)
                    query += f" AND status IN ({status_placeholders})"
                    params.extend([s.value for s in filters.status])
                
                if filters.date_from:
                    query += " AND created_at >= ?"
                    params.append(filters.date_from)
                
                if filters.date_to:
                    query += " AND created_at <= ?"
                    params.append(filters.date_to)
            
            query += " ORDER BY created_at DESC LIMIT 100"
            
            cursor.execute(query, params)
            rows = cursor.fetchall()
            conn.close()
            
            requests = []
            # Convert rows to ServiceCreditRequest objects
            # (Implementation would properly deserialize)
            
            return requests
            
        except Exception as e:
            logger.error(f"Failed to get credit requests: {e}")
            return []
    
    async def get_analytics(self) -> CreditAnalytics:
        """Get credit request analytics"""
        try:
            conn = sqlite3.connect(self.database_path)
            cursor = conn.cursor()
            
            # Get total requests
            cursor.execute("SELECT COUNT(*) FROM service_credit_requests")
            total_requests = cursor.fetchone()[0]
            
            # Get approval rate
            cursor.execute("""
                SELECT COUNT(*) FROM service_credit_requests 
                WHERE status IN ('APPROVED', 'PARTIALLY_APPROVED')
            """)
            approved_requests = cursor.fetchone()[0]
            
            approval_rate = (approved_requests / total_requests * 100) if total_requests > 0 else 0
            
            conn.close()
            
            return CreditAnalytics(
                total_requests=total_requests,
                total_claimed_amount=150000.0,  # Mock data
                total_approved_amount=120000.0,  # Mock data
                approval_rate=approval_rate,
                average_processing_days=8.5,
                requests_by_status={
                    "APPROVED": approved_requests,
                    "REJECTED": total_requests - approved_requests
                },
                requests_by_service={
                    "COMPUTE_ENGINE": 15,
                    "KUBERNETES_ENGINE": 8,
                    "CLOUD_STORAGE": 5
                },
                success_factors=[
                    "Strong monitoring evidence",
                    "Clear SLA violation documentation",
                    "Detailed business impact assessment"
                ],
                improvement_opportunities=[
                    "Faster incident detection",
                    "Better evidence collection",
                    "Proactive SLA monitoring"
                ]
            )
            
        except Exception as e:
            logger.error(f"Failed to get analytics: {e}")
            return CreditAnalytics(
                total_requests=0,
                total_claimed_amount=0.0,
                total_approved_amount=0.0,
                approval_rate=0.0,
                average_processing_days=0.0
            )