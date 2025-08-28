"""
Network Data Models for Networking Troubleshooting Ninja
========================================================

Comprehensive data models for VPC Flow Logs, connectivity testing,
error analysis, and network troubleshooting functionality.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Union
from pydantic import BaseModel, Field


# Enums and Constants
class LogType(str, Enum):
    VPC_FLOW = "VPC_FLOW"
    FIREWALL = "FIREWALL" 
    CLOUD_NAT = "CLOUD_NAT"
    LOAD_BALANCER = "LOAD_BALANCER"
    HTTP_LOAD_BALANCER = "HTTP_LOAD_BALANCER"


class NetworkAction(str, Enum):
    ACCEPT = "ACCEPT"
    DENY = "DENY"
    DROP = "DROP"
    REJECT = "REJECT"


class TestType(str, Enum):
    PING = "PING"
    TRACEROUTE = "TRACEROUTE"
    PORT_SCAN = "PORT_SCAN"
    HTTP_CHECK = "HTTP_CHECK"
    TCP_CONNECT = "TCP_CONNECT"
    UDP_CONNECT = "UDP_CONNECT"


class TestStatus(str, Enum):
    SUCCESS = "SUCCESS"
    FAILURE = "FAILURE"
    TIMEOUT = "TIMEOUT"
    PARTIAL = "PARTIAL"
    IN_PROGRESS = "IN_PROGRESS"
    CANCELLED = "CANCELLED"


class Priority(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class RiskLevel(str, Enum):
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    NONE = "NONE"


# Core Network Models
@dataclass
class NetworkEndpoint:
    """Represents a network endpoint for connectivity testing"""
    type: str  # IP_ADDRESS, INSTANCE, LOAD_BALANCER, etc.
    ip_address: Optional[str] = None
    instance_id: Optional[str] = None
    zone: Optional[str] = None
    port: Optional[int] = None
    protocol: str = "TCP"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "type": self.type,
            "ip_address": self.ip_address,
            "instance_id": self.instance_id,
            "zone": self.zone,
            "port": self.port,
            "protocol": self.protocol
        }


@dataclass
class NetworkHop:
    """Represents a single hop in a network trace"""
    hop_number: int
    ip_address: str
    hostname: Optional[str] = None
    latency_ms: Optional[float] = None
    packet_loss_percent: Optional[float] = None
    timeout: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hop_number": self.hop_number,
            "ip_address": self.ip_address,
            "hostname": self.hostname,
            "latency_ms": self.latency_ms,
            "packet_loss_percent": self.packet_loss_percent,
            "timeout": self.timeout
        }


class NetworkLogEntry(BaseModel):
    """VPC Flow Log entry with enriched metadata"""
    timestamp: datetime
    log_type: LogType
    source_ip: str
    destination_ip: str
    source_port: Optional[int] = None
    destination_port: Optional[int] = None
    protocol: str = "TCP"
    action: NetworkAction
    bytes_sent: int = 0
    packets_sent: int = 0
    bytes_received: int = 0
    packets_received: int = 0
    instance_id: Optional[str] = None
    network_name: Optional[str] = None
    subnet_name: Optional[str] = None
    zone: Optional[str] = None
    region: Optional[str] = None
    project_id: str
    
    # Enriched metadata
    source_instance_name: Optional[str] = None
    destination_instance_name: Optional[str] = None
    firewall_rules_applied: List[str] = Field(default_factory=list)
    route_name: Optional[str] = None
    security_score: float = 100.0  # 0-100, lower = more suspicious
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API responses"""
        return {
            "timestamp": self.timestamp.isoformat(),
            "log_type": self.log_type,
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "source_port": self.source_port,
            "destination_port": self.destination_port,
            "protocol": self.protocol,
            "action": self.action,
            "bytes_sent": self.bytes_sent,
            "packets_sent": self.packets_sent,
            "bytes_received": self.bytes_received,
            "packets_received": self.packets_received,
            "instance_id": self.instance_id,
            "network_name": self.network_name,
            "subnet_name": self.subnet_name,
            "zone": self.zone,
            "region": self.region,
            "project_id": self.project_id,
            "source_instance_name": self.source_instance_name,
            "destination_instance_name": self.destination_instance_name,
            "firewall_rules_applied": self.firewall_rules_applied,
            "route_name": self.route_name,
            "security_score": self.security_score
        }


class ConnectivityTestResult(BaseModel):
    """Result of a network connectivity test"""
    test_id: str
    source: NetworkEndpoint
    destination: NetworkEndpoint
    test_type: TestType
    status: TestStatus
    latency_ms: Optional[float] = None
    packet_loss_percent: Optional[float] = None
    hop_details: List[NetworkHop] = Field(default_factory=list)
    error_message: Optional[str] = None
    timestamp: datetime
    duration_ms: Optional[int] = None
    additional_info: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
    
    @property
    def is_successful(self) -> bool:
        return self.status == TestStatus.SUCCESS
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_id": self.test_id,
            "source": self.source.to_dict(),
            "destination": self.destination.to_dict(),
            "test_type": self.test_type,
            "status": self.status,
            "latency_ms": self.latency_ms,
            "packet_loss_percent": self.packet_loss_percent,
            "hop_details": [hop.to_dict() for hop in self.hop_details],
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "duration_ms": self.duration_ms,
            "additional_info": self.additional_info,
            "is_successful": self.is_successful
        }


# Traffic Pattern Analysis Models
@dataclass
class TrafficPattern:
    """Network traffic pattern analysis"""
    source_ip: str
    destination_ip: str
    protocol: str
    bytes_transferred: int
    packet_count: int
    duration: timedelta
    security_score: float  # 0-100, lower = more suspicious
    unique_ports: int = 0
    flow_count: int = 1
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source_ip": self.source_ip,
            "destination_ip": self.destination_ip,
            "protocol": self.protocol,
            "bytes_transferred": self.bytes_transferred,
            "packet_count": self.packet_count,
            "duration_seconds": self.duration.total_seconds(),
            "security_score": self.security_score,
            "unique_ports": self.unique_ports,
            "flow_count": self.flow_count,
            "first_seen": self.first_seen.isoformat() if self.first_seen else None,
            "last_seen": self.last_seen.isoformat() if self.last_seen else None
        }


class NetworkAnomaly(BaseModel):
    """Detected network anomaly"""
    anomaly_id: str
    anomaly_type: str  # UNUSUAL_TRAFFIC, PORT_SCAN, DDoS, etc.
    severity: RiskLevel
    confidence_score: float  # 0-1
    description: str
    affected_ips: List[str] = Field(default_factory=list)
    detection_time: datetime
    evidence: List[str] = Field(default_factory=list)
    recommended_actions: List[str] = Field(default_factory=list)
    related_log_entries: List[str] = Field(default_factory=list)  # Log entry IDs
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "anomaly_id": self.anomaly_id,
            "anomaly_type": self.anomaly_type,
            "severity": self.severity,
            "confidence_score": self.confidence_score,
            "description": self.description,
            "affected_ips": self.affected_ips,
            "detection_time": self.detection_time.isoformat(),
            "evidence": self.evidence,
            "recommended_actions": self.recommended_actions,
            "related_log_entries": self.related_log_entries
        }


# Route Analysis Models
@dataclass
class Route:
    """Network route information"""
    route_name: str
    destination_range: str
    priority: int
    next_hop_type: str  # instance, ip, gateway, etc.
    next_hop_value: str
    network: str
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class RoutePath:
    """Complete routing path between endpoints"""
    source: NetworkEndpoint
    destination: NetworkEndpoint
    hops: List[NetworkHop]
    total_latency_ms: float
    routes_used: List[Route]
    is_optimal: bool = True
    potential_issues: List[str] = None
    
    def __post_init__(self):
        if self.potential_issues is None:
            self.potential_issues = []


class RoutingAnalysis(BaseModel):
    """Analysis of routing configuration"""
    network_name: str
    total_routes: int
    custom_routes: int
    default_routes: int
    route_conflicts: List[Dict[str, Any]] = Field(default_factory=list)
    optimization_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    routing_loops: List[Dict[str, Any]] = Field(default_factory=list)
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


# Firewall Analysis Models
@dataclass
class FirewallRule:
    """Firewall rule information"""
    rule_name: str
    direction: str  # INGRESS, EGRESS
    priority: int
    action: str  # ALLOW, DENY
    source_ranges: List[str]
    destination_ranges: List[str]
    protocols: List[Dict[str, Any]]  # [{protocol: TCP, ports: [80, 443]}]
    target_tags: List[str] = None
    source_tags: List[str] = None
    service_accounts: List[str] = None
    
    def __post_init__(self):
        if self.target_tags is None:
            self.target_tags = []
        if self.source_tags is None:
            self.source_tags = []
        if self.service_accounts is None:
            self.service_accounts = []


@dataclass
class RuleConflict:
    """Firewall rule conflict detection"""
    primary_rule: str
    conflicting_rule: str
    conflict_type: str  # SHADOW, REDUNDANT, CONTRADICTION
    description: str
    severity: RiskLevel
    recommendation: str


class FirewallAnalysis(BaseModel):
    """Comprehensive firewall analysis"""
    network_name: str
    total_rules: int
    ingress_rules: int
    egress_rules: int
    rule_conflicts: List[RuleConflict] = Field(default_factory=list)
    security_gaps: List[Dict[str, Any]] = Field(default_factory=list)
    optimization_opportunities: List[Dict[str, Any]] = Field(default_factory=list)
    overly_permissive_rules: List[str] = Field(default_factory=list)
    timestamp: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


# Time Range Model for Queries
class TimeRange(BaseModel):
    """Time range for log analysis queries"""
    start: datetime
    end: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }
    
    def duration(self) -> timedelta:
        return self.end - self.start
    
    def duration_hours(self) -> float:
        return self.duration().total_seconds() / 3600
    
    def to_dict(self) -> Dict[str, str]:
        return {
            "start": self.start.isoformat(),
            "end": self.end.isoformat()
        }


# API Request/Response Models
class LogAnalysisRequest(BaseModel):
    """Request model for log analysis"""
    time_range: TimeRange
    filters: Dict[str, Any] = Field(default_factory=dict)
    analysis_type: str = "TRAFFIC_PATTERNS"
    max_results: int = 1000
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


class LogAnalysisResponse(BaseModel):
    """Response model for log analysis"""
    analysis_summary: Dict[str, Any]
    traffic_patterns: List[TrafficPattern] = Field(default_factory=list)
    anomalies: List[NetworkAnomaly] = Field(default_factory=list)
    recommendations: List[str] = Field(default_factory=list)
    total_logs_processed: int
    processing_time_ms: int
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


class ConnectivityTestRequest(BaseModel):
    """Request model for connectivity testing"""
    source: NetworkEndpoint
    destination: NetworkEndpoint
    test_types: List[TestType] = Field(default_factory=lambda: [TestType.PING])
    timeout_seconds: int = 60
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source.to_dict(),
            "destination": self.destination.to_dict(),
            "test_types": self.test_types,
            "timeout_seconds": self.timeout_seconds
        }


# Network Health Models
class NetworkHealth(BaseModel):
    """Overall network health status"""
    overall_score: float  # 0-100
    traffic_health: float
    connectivity_health: float
    security_health: float
    performance_health: float
    issues_detected: int
    critical_issues: int
    recommendations_count: int
    last_updated: datetime
    
    class Config:
        json_encoders = {
            datetime: lambda dt: dt.isoformat(),
        }


# Utility Functions
def create_network_endpoint(ip: str, port: Optional[int] = None) -> NetworkEndpoint:
    """Helper function to create a basic IP-based network endpoint"""
    return NetworkEndpoint(
        type="IP_ADDRESS",
        ip_address=ip,
        port=port
    )


def create_instance_endpoint(instance_id: str, zone: str, port: Optional[int] = None) -> NetworkEndpoint:
    """Helper function to create an instance-based network endpoint"""
    return NetworkEndpoint(
        type="INSTANCE",
        instance_id=instance_id,
        zone=zone,
        port=port
    )


# Export all models for easy importing
__all__ = [
    # Enums
    "LogType", "NetworkAction", "TestType", "TestStatus", "Priority", "RiskLevel",
    
    # Core Models
    "NetworkEndpoint", "NetworkHop", "NetworkLogEntry", "ConnectivityTestResult",
    "TrafficPattern", "NetworkAnomaly",
    
    # Route Models
    "Route", "RoutePath", "RoutingAnalysis",
    
    # Firewall Models
    "FirewallRule", "RuleConflict", "FirewallAnalysis",
    
    # Utility Models
    "TimeRange", "NetworkHealth",
    
    # API Models
    "LogAnalysisRequest", "LogAnalysisResponse", "ConnectivityTestRequest",
    
    # Helper Functions
    "create_network_endpoint", "create_instance_endpoint"
]