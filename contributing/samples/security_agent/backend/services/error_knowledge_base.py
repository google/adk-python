"""
Internal Error Code Knowledge Base Service
=========================================

Comprehensive error code analysis and resolution system with
learning capabilities and contextual recommendations.
"""

import json
import logging
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Set
import uuid

from ..models.error_models import (
    ErrorCodeEntry, ErrorAnalysis, Resolution, ErrorOccurrence,
    ErrorKnowledgeBase, ProbableCause, ImpactAssessment,
    ErrorSeverity, GCPService, ResolutionStatus, ImpactLevel,
    DocumentationLink, ResolutionPattern, EnvironmentalFactor,
    Step, ValidationCheck, RollbackPlan,
    calculate_resolution_priority
)

logger = logging.getLogger(__name__)


class InternalErrorKnowledgeBase:
    """Comprehensive error code analysis and resolution system"""
    
    def __init__(self, database_path: str = "backend/cache/error_knowledge_base.db"):
        """Initialize the error knowledge base"""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Initialize database
        self._init_database()
        
        # Load initial error patterns
        self._populate_initial_errors()
        
        logger.info(f"Initialized Error Knowledge Base at: {database_path}")
    
    def _init_database(self):
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Error codes table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS error_codes (
                        error_code TEXT PRIMARY KEY,
                        service TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        short_description TEXT NOT NULL,
                        detailed_description TEXT,
                        common_causes TEXT,  -- JSON array
                        resolution_patterns TEXT,  -- JSON array
                        related_documentation TEXT,  -- JSON array
                        success_rate REAL DEFAULT 0.0,
                        average_resolution_time INTEGER DEFAULT 3600,  -- seconds
                        occurrence_frequency INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                
                # Error occurrences table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS error_occurrences (
                        occurrence_id TEXT PRIMARY KEY,
                        error_code TEXT NOT NULL,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        context TEXT,  -- JSON
                        resolution_used TEXT,
                        resolution_status TEXT,
                        resolution_time INTEGER,  -- seconds
                        notes TEXT,
                        FOREIGN KEY (error_code) REFERENCES error_codes (error_code)
                    )
                """)
                
                # Resolutions table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS resolutions (
                        resolution_id TEXT PRIMARY KEY,
                        error_code TEXT NOT NULL,
                        title TEXT NOT NULL,
                        description TEXT,
                        resolution_steps TEXT,  -- JSON array
                        estimated_total_time INTEGER DEFAULT 3600,  -- seconds
                        success_rate REAL DEFAULT 0.8,
                        validation_checks TEXT,  -- JSON array
                        rollback_plan TEXT,  -- JSON
                        prerequisites TEXT,  -- JSON array
                        permissions_required TEXT,  -- JSON array
                        tools_required TEXT,  -- JSON array
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        FOREIGN KEY (error_code) REFERENCES error_codes (error_code)
                    )
                """)
                
                # Create indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_error_code ON error_occurrences (error_code)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON error_occurrences (timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_service ON error_codes (service)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_severity ON error_codes (severity)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    def _populate_initial_errors(self):
        """Populate database with initial common GCP errors"""
        try:
            # Check if we already have errors
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM error_codes")
                count = cursor.fetchone()[0]
                
                if count > 0:
                    logger.info(f"Database already contains {count} error codes")
                    return
            
            # Load initial error codes
            initial_errors = self._get_initial_error_codes()
            
            for error_entry in initial_errors:
                self.add_error_entry(error_entry)
            
            logger.info(f"Populated database with {len(initial_errors)} initial error codes")
            
        except Exception as e:
            logger.error(f"Error populating initial errors: {e}")
    
    def _get_initial_error_codes(self) -> List[ErrorCodeEntry]:
        """Get list of initial common GCP error codes"""
        initial_errors = [
            # Networking Errors
            ErrorCodeEntry(
                error_code="NETWORK_UNREACHABLE",
                service=GCPService.NETWORKING,
                severity=ErrorSeverity.HIGH,
                short_description="Network destination is unreachable",
                detailed_description="The target network or host cannot be reached, typically due to routing issues, firewall rules, or network configuration problems.",
                common_causes=[
                    "Firewall rules blocking traffic",
                    "Incorrect routing configuration",
                    "Target instance is down",
                    "Subnet misconfiguration",
                    "VPC peering issues"
                ],
                related_documentation=[
                    DocumentationLink(
                        title="VPC Firewall Rules",
                        url="https://cloud.google.com/vpc/docs/firewalls",
                        type="OFFICIAL"
                    ),
                    DocumentationLink(
                        title="Troubleshooting Network Connectivity",
                        url="https://cloud.google.com/vpc/docs/troubleshooting",
                        type="TROUBLESHOOTING"
                    )
                ]
            ),
            
            ErrorCodeEntry(
                error_code="CONNECTION_REFUSED",
                service=GCPService.NETWORKING,
                severity=ErrorSeverity.MEDIUM,
                short_description="Connection refused by target",
                detailed_description="The target host is reachable but actively refusing connections on the specified port.",
                common_causes=[
                    "Service not running on target port",
                    "Application-level firewall blocking connections",
                    "Incorrect port configuration",
                    "Service binding to localhost only",
                    "Load balancer health check failures"
                ]
            ),
            
            ErrorCodeEntry(
                error_code="TIMEOUT",
                service=GCPService.NETWORKING,
                severity=ErrorSeverity.MEDIUM,
                short_description="Connection or operation timed out",
                detailed_description="A network operation failed to complete within the specified timeout period.",
                common_causes=[
                    "Network latency issues",
                    "Packet loss",
                    "Overloaded target service",
                    "Incorrect timeout settings",
                    "Network congestion"
                ]
            ),
            
            # VPC Errors
            ErrorCodeEntry(
                error_code="SUBNET_NOT_FOUND",
                service=GCPService.VPC,
                severity=ErrorSeverity.HIGH,
                short_description="Specified subnet does not exist",
                detailed_description="The requested subnet is not found in the specified VPC network.",
                common_causes=[
                    "Incorrect subnet name or ID",
                    "Subnet deleted or moved",
                    "Wrong project or region",
                    "Insufficient permissions to access subnet",
                    "Typo in subnet reference"
                ]
            ),
            
            ErrorCodeEntry(
                error_code="IP_EXHAUSTED",
                service=GCPService.VPC,
                severity=ErrorSeverity.HIGH,
                short_description="No available IP addresses in subnet",
                detailed_description="The subnet has run out of available IP addresses for new instances.",
                common_causes=[
                    "Subnet IP range too small",
                    "Too many instances in subnet",
                    "Reserved IP addresses not released",
                    "IP address fragmentation",
                    "Need to expand subnet range"
                ]
            ),
            
            # Firewall Errors
            ErrorCodeEntry(
                error_code="FIREWALL_RULE_DENIED",
                service=GCPService.FIREWALL,
                severity=ErrorSeverity.MEDIUM,
                short_description="Traffic blocked by firewall rule",
                detailed_description="Network traffic was denied by a VPC firewall rule.",
                common_causes=[
                    "Explicit deny rule with higher priority",
                    "No matching allow rule",
                    "Incorrect source/destination tags",
                    "Wrong protocol or port specification",
                    "Service account restrictions"
                ]
            ),
            
            # Compute Engine Errors
            ErrorCodeEntry(
                error_code="INSTANCE_NOT_RUNNING",
                service=GCPService.COMPUTE_ENGINE,
                severity=ErrorSeverity.MEDIUM,
                short_description="Instance is not in running state",
                detailed_description="The target Compute Engine instance is not running and cannot accept connections.",
                common_causes=[
                    "Instance stopped or terminated",
                    "Instance in error state",
                    "Startup script failures",
                    "Resource constraints",
                    "Preemptible instance terminated"
                ]
            ),
            
            # Load Balancer Errors
            ErrorCodeEntry(
                error_code="BACKEND_NOT_HEALTHY",
                service=GCPService.LOAD_BALANCER,
                severity=ErrorSeverity.HIGH,
                short_description="Load balancer backend is unhealthy",
                detailed_description="The load balancer has marked one or more backend instances as unhealthy.",
                common_causes=[
                    "Backend service not responding to health checks",
                    "Incorrect health check configuration",
                    "Backend overloaded",
                    "Application errors",
                    "Network connectivity issues to backends"
                ]
            ),
            
            # Cloud NAT Errors
            ErrorCodeEntry(
                error_code="NAT_ALLOCATION_FAILED",
                service=GCPService.CLOUD_NAT,
                severity=ErrorSeverity.HIGH,
                short_description="Cloud NAT IP allocation failed",
                detailed_description="Cloud NAT was unable to allocate external IP addresses for outbound traffic.",
                common_causes=[
                    "Insufficient external IP addresses",
                    "Regional IP quota exceeded",
                    "NAT gateway configuration issues",
                    "Billing account problems",
                    "IP address allocation conflicts"
                ]
            ),
            
            # DNS Errors
            ErrorCodeEntry(
                error_code="DNS_RESOLUTION_FAILED",
                service=GCPService.CLOUD_DNS,
                severity=ErrorSeverity.MEDIUM,
                short_description="DNS name resolution failed",
                detailed_description="Unable to resolve the specified DNS name to an IP address.",
                common_causes=[
                    "DNS record does not exist",
                    "DNS server unreachable",
                    "Incorrect DNS configuration",
                    "DNS propagation delays",
                    "Network connectivity to DNS servers"
                ]
            )
        ]
        
        return initial_errors
    
    def add_error_entry(self, entry: ErrorCodeEntry):
        """Add or update error entry in the knowledge base"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT OR REPLACE INTO error_codes (
                        error_code, service, severity, short_description, detailed_description,
                        common_causes, resolution_patterns, related_documentation,
                        success_rate, average_resolution_time, occurrence_frequency, last_updated
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    entry.error_code,
                    entry.service.value,
                    entry.severity.value,
                    entry.short_description,
                    entry.detailed_description,
                    json.dumps(entry.common_causes),
                    json.dumps([rp.__dict__ for rp in entry.resolution_patterns]),
                    json.dumps([dl.__dict__ for dl in entry.related_documentation]),
                    entry.success_rate,
                    int(entry.average_resolution_time.total_seconds()),
                    entry.occurrence_frequency,
                    entry.last_updated.isoformat()
                ))
                
                conn.commit()
                logger.debug(f"Added/updated error entry: {entry.error_code}")
                
        except Exception as e:
            logger.error(f"Error adding error entry {entry.error_code}: {e}")
            raise
    
    def get_error_entry(self, error_code: str) -> Optional[ErrorCodeEntry]:
        """Get error entry by error code"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM error_codes WHERE error_code = ?", (error_code,))
                row = cursor.fetchone()
                
                if row:
                    return self._row_to_error_entry(row)
                return None
                
        except Exception as e:
            logger.error(f"Error getting error entry {error_code}: {e}")
            return None
    
    def _row_to_error_entry(self, row) -> ErrorCodeEntry:
        """Convert database row to ErrorCodeEntry"""
        return ErrorCodeEntry(
            error_code=row[0],
            service=GCPService(row[1]),
            severity=ErrorSeverity(row[2]),
            short_description=row[3],
            detailed_description=row[4] or "",
            common_causes=json.loads(row[5]) if row[5] else [],
            resolution_patterns=[ResolutionPattern(**rp) for rp in json.loads(row[6])] if row[6] else [],
            related_documentation=[DocumentationLink(**dl) for dl in json.loads(row[7])] if row[7] else [],
            success_rate=row[8],
            average_resolution_time=timedelta(seconds=row[9]),
            occurrence_frequency=row[10],
            last_updated=datetime.fromisoformat(row[11])
        )
    
    def analyze_error(self, error_code: str, context: Dict[str, Any]) -> ErrorAnalysis:
        """
        Analyze error code with contextual information
        
        Args:
            error_code: The error code to analyze
            context: Contextual information about the error
        
        Returns:
            Comprehensive error analysis
        """
        try:
            logger.info(f"Analyzing error code: {error_code}")
            
            # Get error entry from knowledge base
            error_entry = self.get_error_entry(error_code)
            
            if not error_entry:
                # Create basic analysis for unknown error
                return self._create_unknown_error_analysis(error_code, context)
            
            # Analyze probable causes based on context
            probable_causes = self._analyze_probable_causes(error_entry, context)
            
            # Analyze environmental factors
            environmental_factors = self._analyze_environmental_factors(context)
            
            # Assess impact
            impact_assessment = self._assess_impact(error_entry, context)
            
            # Calculate confidence score
            confidence_score = self._calculate_confidence_score(error_entry, context, probable_causes)
            
            analysis = ErrorAnalysis(
                error_code=error_code,
                original_error_message=context.get('error_message', ''),
                probable_causes=probable_causes,
                environmental_factors=environmental_factors,
                impact_assessment=impact_assessment,
                confidence_score=confidence_score,
                context_data=context
            )
            
            logger.info(f"Completed error analysis for {error_code} with confidence {confidence_score:.2f}")
            return analysis
            
        except Exception as e:
            logger.error(f"Error analyzing error code {error_code}: {e}")
            # Return basic analysis on error
            return ErrorAnalysis(
                error_code=error_code,
                original_error_message=context.get('error_message', ''),
                impact_assessment=ImpactAssessment(impact_level=ImpactLevel.MEDIUM),
                confidence_score=0.1
            )
    
    def _create_unknown_error_analysis(self, error_code: str, context: Dict[str, Any]) -> ErrorAnalysis:
        """Create analysis for unknown error code"""
        return ErrorAnalysis(
            error_code=error_code,
            original_error_message=context.get('error_message', ''),
            probable_causes=[
                ProbableCause(
                    cause_type="UNKNOWN",
                    description="Error code not found in knowledge base",
                    confidence=0.1,
                    evidence=["Error code not documented"],
                    resolution_steps=["Research error code in GCP documentation", "Check GCP status page", "Contact support"]
                )
            ],
            impact_assessment=ImpactAssessment(impact_level=ImpactLevel.MEDIUM),
            confidence_score=0.1
        )
    
    def _analyze_probable_causes(
        self, 
        error_entry: ErrorCodeEntry, 
        context: Dict[str, Any]
    ) -> List[ProbableCause]:
        """Analyze probable causes based on error entry and context"""
        probable_causes = []
        
        try:
            # Create causes from common causes in knowledge base
            for i, cause_desc in enumerate(error_entry.common_causes[:5]):  # Top 5 causes
                confidence = max(0.3, 0.9 - (i * 0.15))  # Decreasing confidence
                
                # Adjust confidence based on context
                confidence = self._adjust_confidence_by_context(cause_desc, context, confidence)
                
                # Generate resolution steps based on cause
                resolution_steps = self._generate_resolution_steps(cause_desc, context)
                
                probable_cause = ProbableCause(
                    cause_type=cause_desc.upper().replace(' ', '_'),
                    description=cause_desc,
                    confidence=confidence,
                    evidence=self._extract_evidence_for_cause(cause_desc, context),
                    resolution_steps=resolution_steps
                )
                probable_causes.append(probable_cause)
            
            # Sort by confidence
            probable_causes.sort(key=lambda c: c.confidence, reverse=True)
            
        except Exception as e:
            logger.warning(f"Error analyzing probable causes: {e}")
        
        return probable_causes
    
    def _adjust_confidence_by_context(
        self, 
        cause_desc: str, 
        context: Dict[str, Any], 
        base_confidence: float
    ) -> float:
        """Adjust confidence score based on contextual clues"""
        try:
            confidence = base_confidence
            
            # Check for context clues that support this cause
            if 'firewall' in cause_desc.lower():
                if 'firewall' in str(context).lower():
                    confidence += 0.2
                if context.get('source_instance') and context.get('destination_ip'):
                    confidence += 0.1
            
            if 'timeout' in cause_desc.lower():
                if 'timeout' in str(context).lower():
                    confidence += 0.3
                if context.get('latency_ms', 0) > 5000:
                    confidence += 0.2
            
            if 'subnet' in cause_desc.lower():
                if 'subnet' in str(context).lower():
                    confidence += 0.2
            
            if 'service' in cause_desc.lower():
                if 'service' in str(context).lower():
                    confidence += 0.1
            
            return min(1.0, confidence)
            
        except Exception as e:
            logger.warning(f"Error adjusting confidence by context: {e}")
            return base_confidence
    
    def _extract_evidence_for_cause(self, cause_desc: str, context: Dict[str, Any]) -> List[str]:
        """Extract evidence supporting a probable cause"""
        evidence = []
        
        try:
            # Add context information as evidence
            if context.get('source_instance'):
                evidence.append(f"Source instance: {context['source_instance']}")
            if context.get('destination_ip'):
                evidence.append(f"Destination IP: {context['destination_ip']}")
            if context.get('timestamp'):
                evidence.append(f"Timestamp: {context['timestamp']}")
            
            # Add specific evidence based on cause type
            if 'firewall' in cause_desc.lower():
                if context.get('firewall_rules'):
                    evidence.append(f"Applied firewall rules: {context['firewall_rules']}")
                if context.get('action') == 'DENY':
                    evidence.append("Traffic was denied by firewall")
            
            if 'timeout' in cause_desc.lower():
                if context.get('latency_ms'):
                    evidence.append(f"Observed latency: {context['latency_ms']}ms")
            
        except Exception as e:
            logger.warning(f"Error extracting evidence: {e}")
        
        return evidence
    
    def _generate_resolution_steps(self, cause_desc: str, context: Dict[str, Any]) -> List[str]:
        """Generate resolution steps based on probable cause"""
        steps = []
        
        try:
            if 'firewall' in cause_desc.lower():
                steps.extend([
                    "Review VPC firewall rules",
                    "Check for explicit deny rules",
                    "Verify source and destination tags",
                    "Test connectivity with firewall rules disabled (if safe)"
                ])
            
            if 'routing' in cause_desc.lower():
                steps.extend([
                    "Check VPC route tables",
                    "Verify custom routes configuration",
                    "Test connectivity between subnets",
                    "Review VPC peering settings"
                ])
            
            if 'instance' in cause_desc.lower():
                steps.extend([
                    "Check instance status",
                    "Verify instance is running",
                    "Review instance startup logs",
                    "Test instance connectivity"
                ])
            
            if 'service' in cause_desc.lower():
                steps.extend([
                    "Check if service is running",
                    "Verify service configuration",
                    "Review service logs",
                    "Test service health endpoints"
                ])
            
            # Add generic steps if no specific ones
            if not steps:
                steps.extend([
                    "Review error logs and context",
                    "Check GCP Console for related alerts",
                    "Verify resource quotas and limits",
                    "Test basic connectivity"
                ])
                
        except Exception as e:
            logger.warning(f"Error generating resolution steps: {e}")
        
        return steps
    
    def _analyze_environmental_factors(self, context: Dict[str, Any]) -> List[EnvironmentalFactor]:
        """Analyze environmental factors that may contribute to the error"""
        factors = []
        
        try:
            # Network configuration factors
            if context.get('vpc_network'):
                factors.append(EnvironmentalFactor(
                    factor_type="NETWORK_CONFIG",
                    description=f"VPC Network: {context['vpc_network']}",
                    impact_score=0.3,
                    evidence=[f"Network: {context['vpc_network']}"]
                ))
            
            if context.get('subnet'):
                factors.append(EnvironmentalFactor(
                    factor_type="SUBNET_CONFIG",
                    description=f"Subnet: {context['subnet']}",
                    impact_score=0.3,
                    evidence=[f"Subnet: {context['subnet']}"]
                ))
            
            # Time-based factors
            current_hour = datetime.now().hour
            if 9 <= current_hour <= 17:  # Business hours
                factors.append(EnvironmentalFactor(
                    factor_type="HIGH_TRAFFIC_PERIOD",
                    description="Error occurred during business hours - higher traffic expected",
                    impact_score=0.2,
                    evidence=[f"Time of day: {current_hour}:00"]
                ))
            
            # Resource factors
            if context.get('instance_id'):
                factors.append(EnvironmentalFactor(
                    factor_type="INSTANCE_SPECIFIC",
                    description=f"Error specific to instance {context['instance_id']}",
                    impact_score=0.4,
                    evidence=[f"Instance ID: {context['instance_id']}"]
                ))
                
        except Exception as e:
            logger.warning(f"Error analyzing environmental factors: {e}")
        
        return factors
    
    def _assess_impact(self, error_entry: ErrorCodeEntry, context: Dict[str, Any]) -> ImpactAssessment:
        """Assess the impact of the error"""
        try:
            # Base impact on error severity
            severity_to_impact = {
                ErrorSeverity.CRITICAL: ImpactLevel.CRITICAL,
                ErrorSeverity.HIGH: ImpactLevel.HIGH,
                ErrorSeverity.MEDIUM: ImpactLevel.MEDIUM,
                ErrorSeverity.LOW: ImpactLevel.LOW,
                ErrorSeverity.INFO: ImpactLevel.NO_IMPACT
            }
            
            base_impact = severity_to_impact.get(error_entry.severity, ImpactLevel.MEDIUM)
            
            # Adjust based on context
            affected_services = []
            affected_users = 0
            
            if context.get('service_name'):
                affected_services.append(context['service_name'])
            
            if context.get('instance_id'):
                affected_services.append(f"Instance {context['instance_id']}")
            
            # Estimate affected users (simplified logic)
            if 'load_balancer' in str(context).lower():
                affected_users = 1000  # Load balancer issues affect many users
            elif 'public' in str(context).lower():
                affected_users = 100   # Public services affect multiple users
            else:
                affected_users = 1     # Internal issues affect fewer users
            
            return ImpactAssessment(
                impact_level=base_impact,
                affected_services=affected_services,
                affected_users=affected_users,
                estimated_downtime=timedelta(minutes=30),  # Default estimate
                sla_impact=base_impact in [ImpactLevel.HIGH, ImpactLevel.CRITICAL]
            )
            
        except Exception as e:
            logger.warning(f"Error assessing impact: {e}")
            return ImpactAssessment(impact_level=ImpactLevel.MEDIUM)
    
    def _calculate_confidence_score(
        self, 
        error_entry: ErrorCodeEntry, 
        context: Dict[str, Any],
        probable_causes: List[ProbableCause]
    ) -> float:
        """Calculate overall confidence score for the analysis"""
        try:
            base_confidence = 0.6  # Base confidence for known errors
            
            # Boost confidence if we have good context
            if context.get('timestamp'):
                base_confidence += 0.1
            if context.get('source_instance') or context.get('destination_ip'):
                base_confidence += 0.1
            if context.get('logs') and len(context['logs']) > 0:
                base_confidence += 0.1
            
            # Factor in probable causes confidence
            if probable_causes:
                avg_cause_confidence = sum(c.confidence for c in probable_causes) / len(probable_causes)
                base_confidence = (base_confidence + avg_cause_confidence) / 2
            
            # Factor in error entry quality
            if len(error_entry.common_causes) > 3:
                base_confidence += 0.05
            if error_entry.success_rate > 0.7:
                base_confidence += 0.05
            
            return min(1.0, base_confidence)
            
        except Exception as e:
            logger.warning(f"Error calculating confidence score: {e}")
            return 0.5
    
    def get_resolution_recommendations(self, error_analysis: ErrorAnalysis) -> List[Resolution]:
        """Generate prioritized resolution recommendations"""
        try:
            logger.info(f"Generating resolution recommendations for {error_analysis.error_code}")
            
            # Get resolutions from database
            db_resolutions = self._get_resolutions_from_db(error_analysis.error_code)
            
            # Generate resolutions from probable causes
            generated_resolutions = self._generate_resolutions_from_causes(error_analysis.probable_causes)
            
            # Combine and deduplicate
            all_resolutions = db_resolutions + generated_resolutions
            
            # Priority rank resolutions
            prioritized_resolutions = calculate_resolution_priority(error_analysis, all_resolutions)
            
            logger.info(f"Generated {len(prioritized_resolutions)} resolution recommendations")
            return prioritized_resolutions[:5]  # Return top 5 recommendations
            
        except Exception as e:
            logger.error(f"Error generating resolution recommendations: {e}")
            return []
    
    def _get_resolutions_from_db(self, error_code: str) -> List[Resolution]:
        """Get existing resolutions from database"""
        resolutions = []
        
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM resolutions WHERE error_code = ?", (error_code,))
                rows = cursor.fetchall()
                
                for row in rows:
                    resolution = Resolution(
                        resolution_id=row[0],
                        title=row[2],
                        description=row[3] or "",
                        resolution_steps=json.loads(row[4]) if row[4] else [],
                        estimated_total_time=timedelta(seconds=row[5]),
                        success_rate=row[6],
                        validation_checks=json.loads(row[7]) if row[7] else [],
                        rollback_plan=json.loads(row[8]) if row[8] else None,
                        prerequisites=json.loads(row[9]) if row[9] else [],
                        permissions_required=json.loads(row[10]) if row[10] else [],
                        tools_required=json.loads(row[11]) if row[11] else []
                    )
                    resolutions.append(resolution)
                    
        except Exception as e:
            logger.warning(f"Error getting resolutions from database: {e}")
        
        return resolutions
    
    def _generate_resolutions_from_causes(self, probable_causes: List[ProbableCause]) -> List[Resolution]:
        """Generate resolutions from probable causes"""
        resolutions = []
        
        try:
            for i, cause in enumerate(probable_causes[:3]):  # Top 3 causes
                if not cause.resolution_steps:
                    continue
                
                resolution = Resolution(
                    resolution_id=str(uuid.uuid4()),
                    title=f"Resolve {cause.cause_type.replace('_', ' ').title()}",
                    description=f"Address the probable cause: {cause.description}",
                    resolution_steps=[
                        Step(
                            step_number=j+1,
                            title=step,
                            description=step,
                            estimated_time=timedelta(minutes=10)
                        ) for j, step in enumerate(cause.resolution_steps)
                    ],
                    success_rate=cause.confidence * 0.8,  # Slightly lower than cause confidence
                    estimated_total_time=timedelta(minutes=len(cause.resolution_steps) * 15)
                )
                resolutions.append(resolution)
                
        except Exception as e:
            logger.warning(f"Error generating resolutions from causes: {e}")
        
        return resolutions
    
    def learn_from_resolution(
        self, 
        error_code: str, 
        resolution: Resolution, 
        success: bool,
        actual_time: Optional[timedelta] = None,
        notes: str = ""
    ):
        """Update knowledge base with resolution feedback"""
        try:
            # Record the occurrence
            occurrence = ErrorOccurrence(
                occurrence_id=str(uuid.uuid4()),
                error_code=error_code,
                timestamp=datetime.now(),
                resolution_used=resolution.resolution_id,
                resolution_status=ResolutionStatus.RESOLVED if success else ResolutionStatus.FAILED,
                resolution_time=actual_time,
                notes=notes
            )
            
            self._record_error_occurrence(occurrence)
            
            # Update resolution success rate
            if success:
                self._update_resolution_success_rate(resolution.resolution_id, True, actual_time)
            else:
                self._update_resolution_success_rate(resolution.resolution_id, False, actual_time)
            
            # Update error entry statistics
            self._update_error_statistics(error_code)
            
            logger.info(f"Learned from resolution: {error_code} - {'Success' if success else 'Failed'}")
            
        except Exception as e:
            logger.error(f"Error learning from resolution: {e}")
    
    def _record_error_occurrence(self, occurrence: ErrorOccurrence):
        """Record error occurrence in database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO error_occurrences (
                        occurrence_id, error_code, timestamp, context, resolution_used,
                        resolution_status, resolution_time, notes
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    occurrence.occurrence_id,
                    occurrence.error_code,
                    occurrence.timestamp.isoformat(),
                    json.dumps(occurrence.context),
                    occurrence.resolution_used,
                    occurrence.resolution_status.value,
                    int(occurrence.resolution_time.total_seconds()) if occurrence.resolution_time else None,
                    occurrence.notes
                ))
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error recording error occurrence: {e}")
    
    def _update_resolution_success_rate(
        self, 
        resolution_id: str, 
        success: bool, 
        actual_time: Optional[timedelta]
    ):
        """Update resolution success rate based on feedback"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Get current statistics
                cursor.execute("SELECT success_rate FROM resolutions WHERE resolution_id = ?", (resolution_id,))
                row = cursor.fetchone()
                
                if row:
                    current_rate = row[0]
                    # Simple moving average (could be improved with more sophisticated method)
                    new_rate = (current_rate * 0.9) + (1.0 if success else 0.0) * 0.1
                    
                    cursor.execute("""
                        UPDATE resolutions 
                        SET success_rate = ? 
                        WHERE resolution_id = ?
                    """, (new_rate, resolution_id))
                    
                    conn.commit()
                    
        except Exception as e:
            logger.error(f"Error updating resolution success rate: {e}")
    
    def _update_error_statistics(self, error_code: str):
        """Update error entry statistics"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Increment occurrence frequency
                cursor.execute("""
                    UPDATE error_codes 
                    SET occurrence_frequency = occurrence_frequency + 1,
                        last_updated = ? 
                    WHERE error_code = ?
                """, (datetime.now().isoformat(), error_code))
                
                conn.commit()
                
        except Exception as e:
            logger.error(f"Error updating error statistics: {e}")


# Export the main class
__all__ = ["InternalErrorKnowledgeBase"]