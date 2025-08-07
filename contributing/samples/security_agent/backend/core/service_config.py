"""Service configuration management for modular backend."""

from typing import Dict, Any, Optional, List
from enum import Enum
from pydantic import BaseModel, Field
import os
import json
import logging

logger = logging.getLogger(__name__)


class ServiceStatus(str, Enum):
    """Service status states."""
    ENABLED = "enabled"
    DISABLED = "disabled"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"
    NOT_CONFIGURED = "not_configured"


class ServiceDependency(BaseModel):
    """Service dependency definition."""
    service_name: str
    required: bool = True
    version: Optional[str] = None


class ServiceHealthCheck(BaseModel):
    """Service health check configuration."""
    endpoint: Optional[str] = None
    interval_seconds: int = 30
    timeout_seconds: int = 5
    failure_threshold: int = 3


class ServiceDefinition(BaseModel):
    """Service definition with all configuration options."""
    name: str
    display_name: str
    description: str
    version: str = "1.0.0"
    enabled_by_default: bool = True
    required: bool = False  # If True, service cannot be disabled
    dependencies: List[ServiceDependency] = Field(default_factory=list)
    health_check: Optional[ServiceHealthCheck] = None
    config: Dict[str, Any] = Field(default_factory=dict)
    api_prefix: Optional[str] = None
    router_module: Optional[str] = None  # e.g., "iam.api"
    service_module: Optional[str] = None  # e.g., "iam.service.IAMService"
    requires_gcp_auth: bool = False
    requires_api_keys: List[str] = Field(default_factory=list)
    tags: List[str] = Field(default_factory=list)


class ServiceConfig:
    """Service configuration manager."""
    
    DEFAULT_CONFIG_PATH = "config/services.json"
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize service configuration."""
        self.config_path = config_path or self.DEFAULT_CONFIG_PATH
        self.services: Dict[str, ServiceDefinition] = {}
        self.runtime_status: Dict[str, ServiceStatus] = {}
        self._load_default_services()
        self._load_config()
    
    def _load_default_services(self):
        """Define default service configurations."""
        default_services = [
            ServiceDefinition(
                name="security",
                display_name="Security Evaluation",
                description="Core security evaluation and scanning service",
                enabled_by_default=True,
                required=True,
                api_prefix="/api/v1/security",
                router_module="security.api",
                service_module="security.service.SecurityService",
                tags=["core", "security"]
            ),
            ServiceDefinition(
                name="iam",
                display_name="IAM Analysis",
                description="Identity and Access Management analysis service",
                enabled_by_default=True,
                api_prefix="/api/v1/iam",
                router_module="iam.api",
                service_module="iam.service.IAMPolicyAnalyzer",
                requires_gcp_auth=True,
                tags=["security", "gcp"]
            ),
            ServiceDefinition(
                name="compliance",
                display_name="Compliance Checking",
                description="Multi-framework compliance evaluation service",
                enabled_by_default=True,
                api_prefix="/api/v1/compliance",
                router_module="compliance.api",
                service_module="compliance.service.ComplianceService",
                requires_gcp_auth=True,
                tags=["compliance", "security"]
            ),
            ServiceDefinition(
                name="cloud_logging",
                display_name="Cloud Logging",
                description="Google Cloud Logging integration service",
                enabled_by_default=True,
                api_prefix="/api/v1/cloud-logging",
                router_module="cloud_logging.api",
                service_module="cloud_logging.service.CloudLoggingService",
                requires_gcp_auth=True,
                dependencies=[ServiceDependency(service_name="gcp", required=True)],
                tags=["logging", "gcp", "monitoring"]
            ),
            ServiceDefinition(
                name="documentation",
                display_name="Documentation Service",
                description="API documentation scraping and analysis",
                enabled_by_default=True,
                api_prefix="/api/v1/documentation",
                router_module="documentation.api",
                service_module="documentation.service.DocumentationService",
                tags=["documentation", "analysis"]
            ),
            ServiceDefinition(
                name="threat_intelligence",
                display_name="Threat Intelligence",
                description="Threat intelligence and vulnerability analysis",
                enabled_by_default=False,
                api_prefix="/api/v1/threat-intelligence",
                router_module="api.threat_intelligence",
                service_module="services.threat_intelligence_service.ThreatIntelligenceService",
                tags=["security", "threats"]
            ),
            ServiceDefinition(
                name="monitoring",
                display_name="Performance Monitoring",
                description="System performance monitoring and metrics",
                enabled_by_default=False,
                api_prefix="/api/v1/monitoring",
                router_module="monitoring.api",
                service_module="monitoring.service.MonitoringService",
                tags=["monitoring", "performance"]
            ),
            ServiceDefinition(
                name="security_analytics",
                display_name="Security Analytics",
                description="BigQuery-based security analytics",
                enabled_by_default=False,
                api_prefix="/api/v1/security-analytics",
                router_module="security_analytics.api",
                service_module="security_analytics.service.SecurityAnalyticsService",
                requires_gcp_auth=True,
                dependencies=[ServiceDependency(service_name="gcp", required=True)],
                tags=["analytics", "security", "bigquery"]
            ),
            ServiceDefinition(
                name="security_knowledge",
                display_name="Security Knowledge Base",
                description="Vertex AI Search integration for security knowledge",
                enabled_by_default=False,
                api_prefix="/api/v1/security-knowledge",
                router_module="security_knowledge.api",
                service_module="security_knowledge.service.SecurityKnowledgeService",
                requires_gcp_auth=True,
                dependencies=[ServiceDependency(service_name="gcp", required=True)],
                tags=["knowledge", "security", "vertex-ai"]
            ),
            ServiceDefinition(
                name="gcp",
                display_name="GCP Core Service",
                description="Core Google Cloud Platform integration service",
                enabled_by_default=True,
                required=True,
                api_prefix="/api/v1/gcp",
                router_module="gcp.api",
                service_module="gcp.service.GCPService",
                requires_gcp_auth=True,
                tags=["core", "gcp"]
            ),
            ServiceDefinition(
                name="agent",
                display_name="AI Agent Service",
                description="Interactive AI security agent",
                enabled_by_default=True,
                api_prefix="/api/v1/agent",
                router_module="api.agent",
                service_module="services.agent_service.AgentService",
                tags=["ai", "agent"]
            ),
            ServiceDefinition(
                name="msa",
                display_name="MSA Analysis",
                description="Microsoft Service Agreement parsing and analysis",
                enabled_by_default=False,
                api_prefix="/api/v1/msa",
                router_module="msa.api",
                service_module="msa.service.MSAParsingService",
                tags=["analysis", "msa"]
            ),
            ServiceDefinition(
                name="tracing",
                display_name="OpenTelemetry Tracing",
                description="Distributed tracing with Cloud Trace",
                enabled_by_default=False,
                api_prefix="/api/v1/tracing",
                router_module="tracing.api",
                service_module="tracing.service.TracingService",
                requires_gcp_auth=True,
                tags=["monitoring", "tracing"]
            ),
            ServiceDefinition(
                name="incident_response",
                display_name="Incident Response",
                description="Security incident management and response",
                enabled_by_default=False,
                api_prefix="/api/v1/incidents",
                router_module="api.incidents",
                service_module="services.incident_response_service.IncidentResponseService",
                tags=["security", "incidents"]
            ),
            ServiceDefinition(
                name="apihub",
                display_name="API Hub Integration",
                description="Google API Hub integration and management",
                enabled_by_default=False,
                api_prefix="/api/v1/apihub",
                router_module="apihub.api",
                service_module="apihub.service.APIHubService",
                requires_gcp_auth=True,
                tags=["integration", "apihub"]
            ),
            ServiceDefinition(
                name="recommendations",
                display_name="Security Recommendations",
                description="AI-powered security recommendations",
                enabled_by_default=True,
                api_prefix="/api/v1/recommendations",
                router_module="recommendations.api",
                service_module="recommendations.service.RecommendationsService",
                dependencies=[ServiceDependency(service_name="security", required=True)],
                tags=["security", "ai", "recommendations"]
            )
        ]
        
        for service in default_services:
            self.services[service.name] = service
            # Set initial runtime status based on enabled_by_default
            self.runtime_status[service.name] = (
                ServiceStatus.NOT_CONFIGURED if service.enabled_by_default 
                else ServiceStatus.DISABLED
            )
    
    def _load_config(self):
        """Load configuration from file if exists."""
        if os.path.exists(self.config_path):
            try:
                with open(self.config_path, 'r') as f:
                    config_data = json.load(f)
                
                # Override default configurations
                for service_name, service_config in config_data.get('services', {}).items():
                    if service_name in self.services:
                        # Update existing service definition
                        service_def = self.services[service_name]
                        for key, value in service_config.items():
                            if hasattr(service_def, key):
                                setattr(service_def, key, value)
                    else:
                        # Add new service definition
                        self.services[service_name] = ServiceDefinition(**service_config)
                
                # Load runtime status
                for service_name, status in config_data.get('runtime_status', {}).items():
                    if service_name in self.services:
                        self.runtime_status[service_name] = ServiceStatus(status)
                        
                logger.info(f"Loaded service configuration from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load service configuration: {e}")
    
    def save_config(self):
        """Save current configuration to file."""
        try:
            config_data = {
                'services': {
                    name: service.model_dump() 
                    for name, service in self.services.items()
                },
                'runtime_status': {
                    name: status.value 
                    for name, status in self.runtime_status.items()
                }
            }
            
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            
            with open(self.config_path, 'w') as f:
                json.dump(config_data, f, indent=2)
                
            logger.info(f"Saved service configuration to {self.config_path}")
        except Exception as e:
            logger.error(f"Failed to save service configuration: {e}")
    
    def get_service(self, name: str) -> Optional[ServiceDefinition]:
        """Get service definition by name."""
        return self.services.get(name)
    
    def get_all_services(self) -> Dict[str, ServiceDefinition]:
        """Get all service definitions."""
        return self.services.copy()
    
    def get_enabled_services(self) -> List[ServiceDefinition]:
        """Get list of enabled services."""
        return [
            service for name, service in self.services.items()
            if self.runtime_status.get(name) not in [ServiceStatus.DISABLED, ServiceStatus.ERROR]
        ]
    
    def get_service_status(self, name: str) -> ServiceStatus:
        """Get runtime status of a service."""
        return self.runtime_status.get(name, ServiceStatus.NOT_CONFIGURED)
    
    def set_service_status(self, name: str, status: ServiceStatus):
        """Set runtime status of a service."""
        if name in self.services:
            # Check if service is required and trying to disable
            if status == ServiceStatus.DISABLED and self.services[name].required:
                raise ValueError(f"Cannot disable required service: {name}")
            
            self.runtime_status[name] = status
            self.save_config()
    
    def enable_service(self, name: str):
        """Enable a service."""
        if name in self.services:
            if self.runtime_status.get(name) == ServiceStatus.DISABLED:
                self.runtime_status[name] = ServiceStatus.NOT_CONFIGURED
                self.save_config()
    
    def disable_service(self, name: str):
        """Disable a service."""
        if name in self.services:
            if self.services[name].required:
                raise ValueError(f"Cannot disable required service: {name}")
            
            self.runtime_status[name] = ServiceStatus.DISABLED
            self.save_config()
    
    def get_service_dependencies(self, name: str) -> List[str]:
        """Get list of service dependencies."""
        service = self.services.get(name)
        if not service:
            return []
        
        return [dep.service_name for dep in service.dependencies]
    
    def check_dependencies(self, name: str) -> bool:
        """Check if all required dependencies are satisfied."""
        service = self.services.get(name)
        if not service:
            return False
        
        for dep in service.dependencies:
            if dep.required:
                dep_status = self.runtime_status.get(dep.service_name)
                if dep_status in [ServiceStatus.DISABLED, ServiceStatus.ERROR, None]:
                    return False
        
        return True
    
    def get_services_by_tag(self, tag: str) -> List[ServiceDefinition]:
        """Get services with a specific tag."""
        return [
            service for service in self.services.values()
            if tag in service.tags
        ]