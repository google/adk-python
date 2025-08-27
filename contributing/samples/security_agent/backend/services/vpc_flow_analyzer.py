"""
VPC Flow Log Analyzer Service
============================

Advanced VPC Flow Log processing with anomaly detection,
traffic pattern analysis, and security scoring.
"""

import asyncio
import logging
import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, AsyncGenerator, Set
from collections import defaultdict, Counter
import statistics

# Optional Google Cloud imports for production use
try:
    from google.cloud import logging as cloud_logging
    from google.cloud import compute_v1
    from google.cloud import monitoring_v3
    GOOGLE_CLOUD_AVAILABLE = True
except ImportError:
    logging.warning("Google Cloud libraries not available - running in test mode")
    GOOGLE_CLOUD_AVAILABLE = False

from ..models.network_models import (
    NetworkLogEntry, TrafficPattern, NetworkAnomaly, TimeRange,
    LogType, NetworkAction, RiskLevel
)

logger = logging.getLogger(__name__)


class VPCFlowLogProcessor:
    """Process and analyze VPC Flow Logs with advanced analytics"""
    
    # Security scoring thresholds
    SUSPICIOUS_PORTS = {22, 23, 25, 53, 80, 135, 139, 445, 993, 995, 1433, 3389, 5432, 5900}
    HIGH_RISK_PROTOCOLS = {"ICMP", "GRE"}
    MAX_BYTES_PER_MINUTE = 1000000000  # 1GB per minute threshold
    MAX_PACKETS_PER_MINUTE = 1000000   # 1M packets per minute threshold
    
    def __init__(self, project_id: str):
        """Initialize VPC Flow Log Processor"""
        self.project_id = project_id
        self.logging_client = cloud_logging.Client(project=project_id)
        self.compute_client = compute_v1.InstancesClient()
        self.monitoring_client = monitoring_v3.MetricServiceClient()
        
        # Caches for performance
        self._instance_cache: Dict[str, Dict] = {}
        self._network_cache: Dict[str, Dict] = {}
        self._cache_expiry = datetime.now() + timedelta(hours=1)
        
        logger.info(f"Initialized VPC Flow Log Processor for project: {project_id}")
    
    def _clear_cache_if_expired(self):
        """Clear caches if they've expired"""
        if datetime.now() > self._cache_expiry:
            self._instance_cache.clear()
            self._network_cache.clear()
            self._cache_expiry = datetime.now() + timedelta(hours=1)
            logger.debug("Cleared expired caches")
    
    async def process_flow_logs(
        self, 
        time_range: TimeRange,
        filters: Optional[Dict[str, Any]] = None,
        max_results: int = 10000
    ) -> List[NetworkLogEntry]:
        """
        Process VPC Flow Logs for the specified time range
        
        Args:
            time_range: Time range for log analysis
            filters: Optional filters for log selection
            max_results: Maximum number of log entries to process
        
        Returns:
            List of processed and enriched network log entries
        """
        self._clear_cache_if_expired()
        
        try:
            logger.info(f"Processing VPC Flow Logs from {time_range.start} to {time_range.end}")
            
            # Build log query
            query = self._build_log_query(time_range, filters)
            logger.debug(f"Log query: {query}")
            
            # Fetch logs from Cloud Logging
            raw_logs = await self._fetch_logs(query, max_results)
            logger.info(f"Fetched {len(raw_logs)} raw log entries")
            
            # Process and enrich logs
            processed_logs = []
            for raw_log in raw_logs:
                try:
                    processed_log = await self._process_single_log(raw_log)
                    if processed_log:
                        processed_logs.append(processed_log)
                except Exception as e:
                    logger.warning(f"Failed to process log entry: {e}")
                    continue
            
            logger.info(f"Successfully processed {len(processed_logs)} log entries")
            return processed_logs
            
        except Exception as e:
            logger.error(f"Error processing VPC Flow Logs: {e}")
            raise
    
    def _build_log_query(self, time_range: TimeRange, filters: Optional[Dict[str, Any]] = None) -> str:
        """Build Cloud Logging query for VPC Flow Logs"""
        
        # Base query for VPC Flow Logs
        query_parts = [
            'resource.type="gce_subnetwork"',
            'logName:"compute.googleapis.com/vpc_flows"'
        ]
        
        # Add time range
        start_time = time_range.start.strftime('%Y-%m-%dT%H:%M:%SZ')
        end_time = time_range.end.strftime('%Y-%m-%dT%H:%M:%SZ')
        query_parts.append(f'timestamp>="{start_time}" timestamp<"{end_time}"')
        
        # Add filters if provided
        if filters:
            if 'source_networks' in filters:
                networks = ' OR '.join([f'resource.labels.subnetwork_name:"{net}"' 
                                      for net in filters['source_networks']])
                query_parts.append(f'({networks})')
            
            if 'protocols' in filters:
                protocols = ' OR '.join([f'jsonPayload.connection.protocol="{proto}"' 
                                       for proto in filters['protocols']])
                query_parts.append(f'({protocols})')
            
            if 'log_types' in filters and 'FIREWALL' in filters['log_types']:
                query_parts.append('(logName:"compute.googleapis.com/firewall" OR logName:"compute.googleapis.com/vpc_flows")')
        
        return ' AND '.join(query_parts)
    
    async def _fetch_logs(self, query: str, max_results: int) -> List[Dict]:
        """Fetch logs from Cloud Logging API"""
        try:
            # Use asyncio.to_thread for blocking call
            entries = await asyncio.to_thread(
                list, 
                self.logging_client.list_entries(filter_=query, max_results=max_results)
            )
            
            return [self._entry_to_dict(entry) for entry in entries]
            
        except Exception as e:
            logger.error(f"Error fetching logs: {e}")
            raise
    
    def _entry_to_dict(self, entry) -> Dict:
        """Convert log entry to dictionary format"""
        try:
            return {
                'timestamp': entry.timestamp,
                'severity': entry.severity,
                'log_name': entry.log_name,
                'resource': dict(entry.resource.labels) if entry.resource else {},
                'json_payload': dict(entry.payload) if hasattr(entry, 'payload') else {},
                'proto_payload': entry.proto_payload if hasattr(entry, 'proto_payload') else None
            }
        except Exception as e:
            logger.warning(f"Error converting log entry to dict: {e}")
            return {}
    
    async def _process_single_log(self, raw_log: Dict) -> Optional[NetworkLogEntry]:
        """Process a single raw log entry into NetworkLogEntry"""
        try:
            # Extract basic information
            timestamp = raw_log.get('timestamp', datetime.now())
            json_payload = raw_log.get('json_payload', {})
            resource_labels = raw_log.get('resource', {})
            
            # Determine log type
            log_name = raw_log.get('log_name', '')
            if 'vpc_flows' in log_name:
                log_type = LogType.VPC_FLOW
            elif 'firewall' in log_name:
                log_type = LogType.FIREWALL
            else:
                log_type = LogType.VPC_FLOW  # Default
            
            # Extract connection information
            connection = json_payload.get('connection', {})
            src_ip = connection.get('src_ip', '')
            dest_ip = connection.get('dest_ip', '')
            src_port = connection.get('src_port')
            dest_port = connection.get('dest_port')
            protocol = connection.get('protocol', 'TCP')
            
            # Extract traffic information
            bytes_sent = connection.get('bytes_sent', 0)
            packets_sent = connection.get('packets_sent', 0)
            
            # Determine action
            action_str = json_payload.get('disposition', 'ACCEPT').upper()
            try:
                action = NetworkAction(action_str)
            except ValueError:
                action = NetworkAction.ACCEPT
            
            # Extract resource information
            project_id = resource_labels.get('project_id', self.project_id)
            subnetwork_name = resource_labels.get('subnetwork_name', '')
            zone = resource_labels.get('zone', '')
            region = zone.rsplit('-', 1)[0] if zone else ''
            
            # Create basic log entry
            log_entry = NetworkLogEntry(
                timestamp=timestamp,
                log_type=log_type,
                source_ip=src_ip,
                destination_ip=dest_ip,
                source_port=src_port,
                destination_port=dest_port,
                protocol=protocol,
                action=action,
                bytes_sent=bytes_sent,
                packets_sent=packets_sent,
                subnet_name=subnetwork_name,
                zone=zone,
                region=region,
                project_id=project_id
            )
            
            # Enrich with metadata
            enriched_entry = await self.enrich_with_metadata([log_entry])
            return enriched_entry[0] if enriched_entry else None
            
        except Exception as e:
            logger.warning(f"Error processing single log entry: {e}")
            return None
    
    async def enrich_with_metadata(self, entries: List[NetworkLogEntry]) -> List[NetworkLogEntry]:
        """
        Enrich log entries with additional metadata
        
        Args:
            entries: List of basic log entries
        
        Returns:
            List of enriched log entries with metadata
        """
        try:
            logger.debug(f"Enriching {len(entries)} log entries with metadata")
            
            # Get unique IPs and zones for batch lookups
            unique_ips = set()
            unique_zones = set()
            for entry in entries:
                unique_ips.update([entry.source_ip, entry.destination_ip])
                if entry.zone:
                    unique_zones.add(entry.zone)
            
            # Batch lookup instance information
            ip_to_instance = await self._batch_lookup_instances(unique_ips, unique_zones)
            
            # Enrich each entry
            enriched_entries = []
            for entry in entries:
                enriched_entry = await self._enrich_single_entry(entry, ip_to_instance)
                enriched_entries.append(enriched_entry)
            
            logger.debug(f"Successfully enriched {len(enriched_entries)} log entries")
            return enriched_entries
            
        except Exception as e:
            logger.error(f"Error enriching log entries: {e}")
            return entries  # Return original entries if enrichment fails
    
    async def _batch_lookup_instances(
        self, 
        ip_addresses: Set[str], 
        zones: Set[str]
    ) -> Dict[str, Dict]:
        """Batch lookup instance information for IP addresses"""
        ip_to_instance = {}
        
        try:
            # Check cache first
            for ip in ip_addresses:
                if ip in self._instance_cache:
                    ip_to_instance[ip] = self._instance_cache[ip]
            
            # Get uncached IPs
            uncached_ips = ip_addresses - set(ip_to_instance.keys())
            
            if uncached_ips and zones:
                # Batch lookup instances in all zones
                for zone in zones:
                    try:
                        instances = await asyncio.to_thread(
                            self.compute_client.list,
                            project=self.project_id,
                            zone=zone
                        )
                        
                        for instance in instances:
                            # Extract instance IPs
                            instance_ips = self._extract_instance_ips(instance)
                            instance_info = {
                                'name': instance.name,
                                'id': str(instance.id),
                                'zone': zone,
                                'status': instance.status,
                                'machine_type': instance.machine_type.split('/')[-1] if instance.machine_type else '',
                                'network_interfaces': []
                            }
                            
                            # Add network interface info
                            for ni in instance.network_interfaces:
                                interface_info = {
                                    'network': ni.network.split('/')[-1] if ni.network else '',
                                    'subnet': ni.subnetwork.split('/')[-1] if ni.subnetwork else '',
                                    'internal_ip': ni.network_i_p if hasattr(ni, 'network_i_p') else '',
                                    'external_ip': ''
                                }
                                
                                # Get external IP if exists
                                if ni.access_configs:
                                    for ac in ni.access_configs:
                                        if hasattr(ac, 'nat_i_p') and ac.nat_i_p:
                                            interface_info['external_ip'] = ac.nat_i_p
                                
                                instance_info['network_interfaces'].append(interface_info)
                            
                            # Map all IPs to this instance
                            for ip in instance_ips:
                                if ip in uncached_ips:
                                    ip_to_instance[ip] = instance_info
                                    self._instance_cache[ip] = instance_info
                    
                    except Exception as e:
                        logger.warning(f"Error looking up instances in zone {zone}: {e}")
                        continue
            
            return ip_to_instance
            
        except Exception as e:
            logger.error(f"Error in batch instance lookup: {e}")
            return {}
    
    def _extract_instance_ips(self, instance) -> List[str]:
        """Extract all IP addresses from an instance"""
        ips = []
        try:
            for ni in instance.network_interfaces:
                # Internal IP
                if hasattr(ni, 'network_i_p') and ni.network_i_p:
                    ips.append(ni.network_i_p)
                
                # External IPs
                if ni.access_configs:
                    for ac in ni.access_configs:
                        if hasattr(ac, 'nat_i_p') and ac.nat_i_p:
                            ips.append(ac.nat_i_p)
        except Exception as e:
            logger.warning(f"Error extracting IPs from instance: {e}")
        
        return ips
    
    async def _enrich_single_entry(
        self, 
        entry: NetworkLogEntry, 
        ip_to_instance: Dict[str, Dict]
    ) -> NetworkLogEntry:
        """Enrich a single log entry with metadata"""
        try:
            # Add instance information
            if entry.source_ip in ip_to_instance:
                instance_info = ip_to_instance[entry.source_ip]
                entry.source_instance_name = instance_info.get('name', '')
                if not entry.instance_id:
                    entry.instance_id = instance_info.get('id', '')
                if not entry.zone:
                    entry.zone = instance_info.get('zone', '')
                if not entry.region and entry.zone:
                    entry.region = entry.zone.rsplit('-', 1)[0]
                
                # Extract network information
                for ni in instance_info.get('network_interfaces', []):
                    if ni.get('internal_ip') == entry.source_ip or ni.get('external_ip') == entry.source_ip:
                        if not entry.network_name:
                            entry.network_name = ni.get('network', '')
                        if not entry.subnet_name:
                            entry.subnet_name = ni.get('subnet', '')
                        break
            
            if entry.destination_ip in ip_to_instance:
                instance_info = ip_to_instance[entry.destination_ip]
                entry.destination_instance_name = instance_info.get('name', '')
            
            # Calculate security score
            entry.security_score = self._calculate_security_score(entry)
            
            return entry
            
        except Exception as e:
            logger.warning(f"Error enriching single entry: {e}")
            return entry
    
    def _calculate_security_score(self, entry: NetworkLogEntry) -> float:
        """
        Calculate security score for a log entry (0-100, lower = more suspicious)
        """
        score = 100.0  # Start with perfect score
        
        try:
            # Check for suspicious ports
            suspicious_ports_penalty = 0
            if entry.source_port and entry.source_port in self.SUSPICIOUS_PORTS:
                suspicious_ports_penalty += 20
            if entry.destination_port and entry.destination_port in self.SUSPICIOUS_PORTS:
                suspicious_ports_penalty += 20
            
            score -= suspicious_ports_penalty
            
            # Check for high-risk protocols
            if entry.protocol in self.HIGH_RISK_PROTOCOLS:
                score -= 15
            
            # Check for denied/dropped traffic (might indicate attack attempts)
            if entry.action in [NetworkAction.DENY, NetworkAction.DROP]:
                score -= 10
            
            # Check for unusual traffic volumes
            if entry.bytes_sent > self.MAX_BYTES_PER_MINUTE:
                score -= 25
            if entry.packets_sent > self.MAX_PACKETS_PER_MINUTE:
                score -= 20
            
            # Check for internal vs external communication
            if self._is_external_ip(entry.destination_ip):
                score -= 5  # External communication is slightly more risky
            
            # Ensure score stays within bounds
            score = max(0.0, min(100.0, score))
            
        except Exception as e:
            logger.warning(f"Error calculating security score: {e}")
            score = 50.0  # Default neutral score
        
        return score
    
    def _is_external_ip(self, ip: str) -> bool:
        """Check if IP address is external (not RFC 1918 private)"""
        try:
            import ipaddress
            ip_obj = ipaddress.ip_address(ip)
            return not ip_obj.is_private
        except:
            return False
    
    async def detect_anomalies(
        self, 
        entries: List[NetworkLogEntry],
        baseline_days: int = 7
    ) -> List[NetworkAnomaly]:
        """
        Detect anomalies in network log entries
        
        Args:
            entries: List of network log entries
            baseline_days: Number of days to use for baseline calculation
        
        Returns:
            List of detected network anomalies
        """
        try:
            logger.info(f"Detecting anomalies in {len(entries)} log entries")
            
            anomalies = []
            
            # Group entries by time windows for analysis
            time_windows = self._group_entries_by_time(entries, window_minutes=5)
            
            # Detect various types of anomalies
            anomalies.extend(await self._detect_traffic_volume_anomalies(time_windows))
            anomalies.extend(await self._detect_port_scan_anomalies(entries))
            anomalies.extend(await self._detect_unusual_traffic_patterns(entries))
            anomalies.extend(await self._detect_security_violations(entries))
            
            logger.info(f"Detected {len(anomalies)} network anomalies")
            return anomalies
            
        except Exception as e:
            logger.error(f"Error detecting anomalies: {e}")
            return []
    
    def _group_entries_by_time(
        self, 
        entries: List[NetworkLogEntry], 
        window_minutes: int = 5
    ) -> Dict[datetime, List[NetworkLogEntry]]:
        """Group log entries by time windows"""
        windows = defaultdict(list)
        
        for entry in entries:
            # Round timestamp to nearest window
            window_start = entry.timestamp.replace(
                minute=(entry.timestamp.minute // window_minutes) * window_minutes,
                second=0,
                microsecond=0
            )
            windows[window_start].append(entry)
        
        return dict(windows)
    
    async def _detect_traffic_volume_anomalies(
        self, 
        time_windows: Dict[datetime, List[NetworkLogEntry]]
    ) -> List[NetworkAnomaly]:
        """Detect anomalies in traffic volume"""
        anomalies = []
        
        try:
            # Calculate statistics for each window
            window_stats = []
            for timestamp, entries in time_windows.items():
                total_bytes = sum(entry.bytes_sent for entry in entries)
                total_packets = sum(entry.packets_sent for entry in entries)
                unique_ips = len(set(entry.source_ip for entry in entries))
                
                window_stats.append({
                    'timestamp': timestamp,
                    'total_bytes': total_bytes,
                    'total_packets': total_packets,
                    'unique_ips': unique_ips,
                    'entry_count': len(entries)
                })
            
            if len(window_stats) < 3:
                return anomalies  # Need at least 3 windows for statistical analysis
            
            # Calculate baseline statistics
            byte_values = [stat['total_bytes'] for stat in window_stats]
            packet_values = [stat['total_packets'] for stat in window_stats]
            
            byte_mean = statistics.mean(byte_values)
            byte_stdev = statistics.stdev(byte_values) if len(byte_values) > 1 else 0
            
            packet_mean = statistics.mean(packet_values)
            packet_stdev = statistics.stdev(packet_values) if len(packet_values) > 1 else 0
            
            # Detect anomalies (values > 3 standard deviations from mean)
            for stat in window_stats:
                if byte_stdev > 0 and abs(stat['total_bytes'] - byte_mean) > 3 * byte_stdev:
                    anomaly = NetworkAnomaly(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type="UNUSUAL_TRAFFIC_VOLUME",
                        severity=RiskLevel.HIGH if stat['total_bytes'] > byte_mean else RiskLevel.MEDIUM,
                        confidence_score=0.85,
                        description=f"Unusual traffic volume detected: {stat['total_bytes']} bytes (baseline: {byte_mean:.0f})",
                        detection_time=stat['timestamp'],
                        evidence=[
                            f"Traffic volume: {stat['total_bytes']} bytes",
                            f"Baseline mean: {byte_mean:.0f} bytes",
                            f"Standard deviation: {byte_stdev:.0f}"
                        ],
                        recommended_actions=[
                            "Review traffic patterns for this time period",
                            "Check for potential DDoS or data exfiltration",
                            "Validate legitimate high-volume transfers"
                        ]
                    )
                    anomalies.append(anomaly)
                
                if packet_stdev > 0 and abs(stat['total_packets'] - packet_mean) > 3 * packet_stdev:
                    anomaly = NetworkAnomaly(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type="UNUSUAL_PACKET_COUNT",
                        severity=RiskLevel.MEDIUM,
                        confidence_score=0.80,
                        description=f"Unusual packet count detected: {stat['total_packets']} packets (baseline: {packet_mean:.0f})",
                        detection_time=stat['timestamp'],
                        evidence=[
                            f"Packet count: {stat['total_packets']}",
                            f"Baseline mean: {packet_mean:.0f}",
                            f"Standard deviation: {packet_stdev:.0f}"
                        ],
                        recommended_actions=[
                            "Investigate source of high packet volume",
                            "Check for potential network scanning or flooding"
                        ]
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.warning(f"Error detecting traffic volume anomalies: {e}")
        
        return anomalies
    
    async def _detect_port_scan_anomalies(self, entries: List[NetworkLogEntry]) -> List[NetworkAnomaly]:
        """Detect potential port scanning activities"""
        anomalies = []
        
        try:
            # Group by source IP and count unique destination ports
            source_port_counts = defaultdict(set)
            source_entries = defaultdict(list)
            
            for entry in entries:
                if entry.destination_port:
                    source_port_counts[entry.source_ip].add(entry.destination_port)
                    source_entries[entry.source_ip].append(entry)
            
            # Detect sources connecting to many different ports (potential port scan)
            for source_ip, ports in source_port_counts.items():
                if len(ports) > 20:  # Threshold for port scanning
                    entries_for_source = source_entries[source_ip]
                    denied_attempts = sum(1 for e in entries_for_source if e.action == NetworkAction.DENY)
                    
                    severity = RiskLevel.HIGH if denied_attempts > len(ports) * 0.5 else RiskLevel.MEDIUM
                    
                    anomaly = NetworkAnomaly(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type="PORT_SCAN",
                        severity=severity,
                        confidence_score=0.90,
                        description=f"Potential port scanning detected from {source_ip}: {len(ports)} unique ports targeted",
                        affected_ips=[source_ip],
                        detection_time=datetime.now(),
                        evidence=[
                            f"Source IP: {source_ip}",
                            f"Unique ports targeted: {len(ports)}",
                            f"Total attempts: {len(entries_for_source)}",
                            f"Denied attempts: {denied_attempts}",
                            f"Ports: {sorted(list(ports))[:20]}"  # Show first 20 ports
                        ],
                        recommended_actions=[
                            f"Block or rate-limit traffic from {source_ip}",
                            "Review firewall rules for the targeted ports",
                            "Monitor for additional scanning activity",
                            "Consider implementing intrusion detection"
                        ]
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.warning(f"Error detecting port scan anomalies: {e}")
        
        return anomalies
    
    async def _detect_unusual_traffic_patterns(self, entries: List[NetworkLogEntry]) -> List[NetworkAnomaly]:
        """Detect unusual traffic patterns"""
        anomalies = []
        
        try:
            # Analyze traffic by protocol distribution
            protocol_counts = Counter(entry.protocol for entry in entries)
            total_entries = len(entries)
            
            # Detect unusual protocol usage
            for protocol, count in protocol_counts.items():
                percentage = (count / total_entries) * 100
                
                # Flag protocols that are >50% of all traffic (unusual for most environments)
                if percentage > 50 and protocol != "TCP":  # TCP dominance is normal
                    anomaly = NetworkAnomaly(
                        anomaly_id=str(uuid.uuid4()),
                        anomaly_type="UNUSUAL_PROTOCOL_DISTRIBUTION",
                        severity=RiskLevel.MEDIUM,
                        confidence_score=0.75,
                        description=f"Unusual protocol distribution: {protocol} accounts for {percentage:.1f}% of traffic",
                        detection_time=datetime.now(),
                        evidence=[
                            f"Protocol: {protocol}",
                            f"Percentage of traffic: {percentage:.1f}%",
                            f"Total occurrences: {count}",
                            f"Expected: TCP should dominate most traffic"
                        ],
                        recommended_actions=[
                            f"Investigate high usage of {protocol} protocol",
                            "Verify if this traffic pattern is legitimate",
                            "Check for potential protocol-specific attacks"
                        ]
                    )
                    anomalies.append(anomaly)
        
        except Exception as e:
            logger.warning(f"Error detecting unusual traffic patterns: {e}")
        
        return anomalies
    
    async def _detect_security_violations(self, entries: List[NetworkLogEntry]) -> List[NetworkAnomaly]:
        """Detect security violations and policy breaches"""
        anomalies = []
        
        try:
            # Count entries with low security scores
            low_security_entries = [e for e in entries if e.security_score < 50.0]
            
            if len(low_security_entries) > len(entries) * 0.1:  # >10% low security score
                # Group by source IP to identify problematic sources
                source_violations = defaultdict(list)
                for entry in low_security_entries:
                    source_violations[entry.source_ip].append(entry)
                
                # Create anomaly for each problematic source
                for source_ip, violation_entries in source_violations.items():
                    if len(violation_entries) > 5:  # Threshold for violations
                        avg_security_score = sum(e.security_score for e in violation_entries) / len(violation_entries)
                        
                        anomaly = NetworkAnomaly(
                            anomaly_id=str(uuid.uuid4()),
                            anomaly_type="SECURITY_POLICY_VIOLATIONS",
                            severity=RiskLevel.HIGH if avg_security_score < 25.0 else RiskLevel.MEDIUM,
                            confidence_score=0.80,
                            description=f"Multiple security violations detected from {source_ip}: average security score {avg_security_score:.1f}",
                            affected_ips=[source_ip],
                            detection_time=datetime.now(),
                            evidence=[
                                f"Source IP: {source_ip}",
                                f"Violation count: {len(violation_entries)}",
                                f"Average security score: {avg_security_score:.1f}",
                                f"Violation types: {set(e.action.value for e in violation_entries)}"
                            ],
                            recommended_actions=[
                                f"Investigate traffic from {source_ip}",
                                "Review firewall rules and security policies",
                                "Consider blocking or rate-limiting suspicious sources",
                                "Implement additional monitoring for this IP"
                            ]
                        )
                        anomalies.append(anomaly)
        
        except Exception as e:
            logger.warning(f"Error detecting security violations: {e}")
        
        return anomalies
    
    async def analyze_traffic_patterns(self, entries: List[NetworkLogEntry]) -> List[TrafficPattern]:
        """
        Analyze traffic patterns from log entries
        
        Args:
            entries: List of network log entries
        
        Returns:
            List of identified traffic patterns
        """
        try:
            logger.info(f"Analyzing traffic patterns from {len(entries)} log entries")
            
            # Group entries by source-destination pairs
            pattern_groups = defaultdict(list)
            for entry in entries:
                key = (entry.source_ip, entry.destination_ip, entry.protocol)
                pattern_groups[key].append(entry)
            
            patterns = []
            for (source_ip, dest_ip, protocol), group_entries in pattern_groups.items():
                if len(group_entries) < 2:  # Skip single-occurrence patterns
                    continue
                
                # Calculate pattern statistics
                total_bytes = sum(e.bytes_sent for e in group_entries)
                total_packets = sum(e.packets_sent for e in group_entries)
                unique_ports = len(set(e.destination_port for e in group_entries if e.destination_port))
                
                # Calculate duration
                timestamps = [e.timestamp for e in group_entries]
                duration = max(timestamps) - min(timestamps)
                
                # Calculate security score (average of all entries)
                avg_security_score = sum(e.security_score for e in group_entries) / len(group_entries)
                
                pattern = TrafficPattern(
                    source_ip=source_ip,
                    destination_ip=dest_ip,
                    protocol=protocol,
                    bytes_transferred=total_bytes,
                    packet_count=total_packets,
                    duration=duration,
                    security_score=avg_security_score,
                    unique_ports=unique_ports,
                    flow_count=len(group_entries),
                    first_seen=min(timestamps),
                    last_seen=max(timestamps)
                )
                patterns.append(pattern)
            
            # Sort by bytes transferred (most active patterns first)
            patterns.sort(key=lambda p: p.bytes_transferred, reverse=True)
            
            logger.info(f"Identified {len(patterns)} traffic patterns")
            return patterns
            
        except Exception as e:
            logger.error(f"Error analyzing traffic patterns: {e}")
            return []


# Export the main class
__all__ = ["VPCFlowLogProcessor"]