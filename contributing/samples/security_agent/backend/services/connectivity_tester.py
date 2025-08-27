"""
Basic Connectivity Testing Framework
===================================

Network connectivity testing service with ping, traceroute, and port scanning
capabilities for comprehensive network troubleshooting.
"""

import asyncio
import logging
import socket
import subprocess
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import platform
import json
import sqlite3
from pathlib import Path

from ..models.network_models import (
    ConnectivityTestResult, NetworkEndpoint, NetworkHop,
    TestType, TestStatus, create_network_endpoint, create_instance_endpoint
)

logger = logging.getLogger(__name__)


class ConnectivityTester:
    """Network connectivity testing service"""
    
    def __init__(self, database_path: str = "backend/cache/connectivity_tests.db"):
        """Initialize connectivity tester with result storage"""
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Platform-specific commands
        self.is_windows = platform.system().lower() == "windows"
        self.ping_cmd = "ping" if self.is_windows else "ping"
        self.traceroute_cmd = "tracert" if self.is_windows else "traceroute"
        
        # Initialize database
        self._init_database()
        
        logger.info(f"Initialized ConnectivityTester with database: {database_path}")
    
    def _init_database(self):
        """Initialize SQLite database for storing test results"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                # Test results table
                cursor.execute("""
                    CREATE TABLE IF NOT EXISTS connectivity_tests (
                        test_id TEXT PRIMARY KEY,
                        source_info TEXT NOT NULL,  -- JSON
                        destination_info TEXT NOT NULL,  -- JSON
                        test_type TEXT NOT NULL,
                        status TEXT NOT NULL,
                        latency_ms REAL,
                        packet_loss_percent REAL,
                        hop_details TEXT,  -- JSON array
                        error_message TEXT,
                        timestamp TEXT NOT NULL,
                        duration_ms INTEGER,
                        additional_info TEXT  -- JSON
                    )
                """)
                
                # Create indexes for performance
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON connectivity_tests (timestamp)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_test_type ON connectivity_tests (test_type)")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_status ON connectivity_tests (status)")
                
                conn.commit()
                logger.info("Database initialized successfully")
                
        except Exception as e:
            logger.error(f"Error initializing database: {e}")
            raise
    
    async def ping_test(
        self, 
        destination: NetworkEndpoint,
        count: int = 4,
        timeout: int = 5
    ) -> ConnectivityTestResult:
        """
        Perform ping connectivity test
        
        Args:
            destination: Target endpoint to ping
            count: Number of ping packets to send
            timeout: Timeout in seconds per ping
            
        Returns:
            Connectivity test result with latency and packet loss info
        """
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Starting ping test {test_id} to {destination.ip_address}")
            
            # Build ping command
            if self.is_windows:
                cmd = ["ping", "-n", str(count), "-w", str(timeout * 1000), destination.ip_address]
            else:
                cmd = ["ping", "-c", str(count), "-W", str(timeout), destination.ip_address]
            
            # Execute ping command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            stdout_str = stdout.decode('utf-8', errors='ignore')
            stderr_str = stderr.decode('utf-8', errors='ignore')
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Parse ping results
            latency_ms, packet_loss_percent = self._parse_ping_output(stdout_str)
            
            # Determine test status
            if process.returncode == 0 and packet_loss_percent < 100:
                status = TestStatus.SUCCESS
                error_message = None
            elif packet_loss_percent == 100:
                status = TestStatus.FAILURE
                error_message = "100% packet loss - destination unreachable"
            else:
                status = TestStatus.FAILURE
                error_message = stderr_str or "Ping command failed"
            
            result = ConnectivityTestResult(
                test_id=test_id,
                source=create_network_endpoint("127.0.0.1"),  # Local machine
                destination=destination,
                test_type=TestType.PING,
                status=status,
                latency_ms=latency_ms,
                packet_loss_percent=packet_loss_percent,
                error_message=error_message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                additional_info={
                    "ping_count": count,
                    "timeout_seconds": timeout,
                    "command_output": stdout_str,
                    "command_error": stderr_str
                }
            )
            
            # Store result in database
            await self._store_test_result(result)
            
            logger.info(f"Ping test {test_id} completed: {status} (latency: {latency_ms}ms, loss: {packet_loss_percent}%)")
            return result
            
        except Exception as e:
            logger.error(f"Error in ping test {test_id}: {e}")
            
            result = ConnectivityTestResult(
                test_id=test_id,
                source=create_network_endpoint("127.0.0.1"),
                destination=destination,
                test_type=TestType.PING,
                status=TestStatus.FAILURE,
                error_message=str(e),
                timestamp=datetime.now(),
                duration_ms=int((time.time() - start_time) * 1000)
            )
            
            await self._store_test_result(result)
            return result
    
    def _parse_ping_output(self, output: str) -> Tuple[Optional[float], float]:
        """Parse ping command output to extract latency and packet loss"""
        try:
            latency_ms = None
            packet_loss_percent = 100.0  # Default to 100% loss
            
            lines = output.split('\n')
            
            if self.is_windows:
                # Windows ping output parsing
                for line in lines:
                    if "time=" in line.lower():
                        # Extract latency from "time=XXXms" or "time<1ms"
                        time_part = line.split("time=")[-1] if "time=" in line else line.split("time<")[-1]
                        time_str = time_part.split("ms")[0].replace("<", "")
                        try:
                            if "<" in line:
                                latency_ms = 0.5  # Less than 1ms
                            else:
                                latency_ms = float(time_str)
                        except ValueError:
                            pass
                    
                    if "lost" in line.lower() and "(" in line and "%" in line:
                        # Extract packet loss percentage
                        try:
                            loss_part = line.split("(")[1].split("%")[0]
                            packet_loss_percent = float(loss_part)
                        except (IndexError, ValueError):
                            pass
            else:
                # Unix/Linux ping output parsing
                for line in lines:
                    if "time=" in line:
                        # Extract latency from "time=XXX ms"
                        try:
                            time_part = line.split("time=")[1].split()[0]
                            latency_ms = float(time_part)
                        except (IndexError, ValueError):
                            pass
                    
                    if "packet loss" in line.lower():
                        # Extract packet loss percentage
                        try:
                            loss_part = line.split("%")[0].split()[-1]
                            packet_loss_percent = float(loss_part)
                        except (IndexError, ValueError):
                            pass
            
            return latency_ms, packet_loss_percent
            
        except Exception as e:
            logger.warning(f"Error parsing ping output: {e}")
            return None, 100.0
    
    async def port_connectivity_test(
        self,
        destination: NetworkEndpoint,
        timeout: int = 10
    ) -> ConnectivityTestResult:
        """
        Test TCP port connectivity
        
        Args:
            destination: Target endpoint with IP and port
            timeout: Connection timeout in seconds
            
        Returns:
            Connectivity test result
        """
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Starting port connectivity test {test_id} to {destination.ip_address}:{destination.port}")
            
            if not destination.port:
                raise ValueError("Port must be specified for port connectivity test")
            
            # Test TCP connection
            conn_start = time.time()
            try:
                # Create socket connection
                sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                sock.settimeout(timeout)
                
                result = sock.connect_ex((destination.ip_address, destination.port))
                conn_time = (time.time() - conn_start) * 1000  # Convert to milliseconds
                
                sock.close()
                
                if result == 0:
                    status = TestStatus.SUCCESS
                    error_message = None
                    latency_ms = conn_time
                else:
                    status = TestStatus.FAILURE
                    error_message = f"Connection refused or timeout (error code: {result})"
                    latency_ms = None
                
            except socket.timeout:
                status = TestStatus.TIMEOUT
                error_message = f"Connection timeout after {timeout} seconds"
                latency_ms = None
            except Exception as conn_error:
                status = TestStatus.FAILURE
                error_message = f"Connection error: {conn_error}"
                latency_ms = None
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            result = ConnectivityTestResult(
                test_id=test_id,
                source=create_network_endpoint("127.0.0.1"),
                destination=destination,
                test_type=TestType.TCP_CONNECT,
                status=status,
                latency_ms=latency_ms,
                error_message=error_message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                additional_info={
                    "timeout_seconds": timeout,
                    "protocol": "TCP"
                }
            )
            
            await self._store_test_result(result)
            
            logger.info(f"Port connectivity test {test_id} completed: {status}")
            return result
            
        except Exception as e:
            logger.error(f"Error in port connectivity test {test_id}: {e}")
            
            result = ConnectivityTestResult(
                test_id=test_id,
                source=create_network_endpoint("127.0.0.1"),
                destination=destination,
                test_type=TestType.TCP_CONNECT,
                status=TestStatus.FAILURE,
                error_message=str(e),
                timestamp=datetime.now(),
                duration_ms=int((time.time() - start_time) * 1000)
            )
            
            await self._store_test_result(result)
            return result
    
    async def traceroute_test(
        self,
        destination: NetworkEndpoint,
        max_hops: int = 30,
        timeout: int = 60
    ) -> ConnectivityTestResult:
        """
        Perform traceroute test to trace network path
        
        Args:
            destination: Target endpoint
            max_hops: Maximum number of hops to trace
            timeout: Total timeout for traceroute command
            
        Returns:
            Connectivity test result with hop details
        """
        test_id = str(uuid.uuid4())
        start_time = time.time()
        
        try:
            logger.info(f"Starting traceroute test {test_id} to {destination.ip_address}")
            
            # Build traceroute command
            if self.is_windows:
                cmd = ["tracert", "-h", str(max_hops), "-w", "5000", destination.ip_address]
            else:
                cmd = ["traceroute", "-m", str(max_hops), "-w", "5", destination.ip_address]
            
            # Execute traceroute command
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(
                    process.communicate(), 
                    timeout=timeout
                )
                stdout_str = stdout.decode('utf-8', errors='ignore')
                stderr_str = stderr.decode('utf-8', errors='ignore')
            except asyncio.TimeoutError:
                process.kill()
                stdout_str = ""
                stderr_str = f"Traceroute timeout after {timeout} seconds"
            
            duration_ms = int((time.time() - start_time) * 1000)
            
            # Parse traceroute output to extract hops
            hop_details = self._parse_traceroute_output(stdout_str)
            
            # Determine test status
            if process.returncode == 0 and hop_details:
                status = TestStatus.SUCCESS
                error_message = None
            elif not hop_details:
                status = TestStatus.FAILURE
                error_message = "No route to destination found"
            else:
                status = TestStatus.PARTIAL
                error_message = stderr_str or "Traceroute completed with issues"
            
            result = ConnectivityTestResult(
                test_id=test_id,
                source=create_network_endpoint("127.0.0.1"),
                destination=destination,
                test_type=TestType.TRACEROUTE,
                status=status,
                hop_details=hop_details,
                error_message=error_message,
                timestamp=datetime.now(),
                duration_ms=duration_ms,
                additional_info={
                    "max_hops": max_hops,
                    "timeout_seconds": timeout,
                    "command_output": stdout_str,
                    "command_error": stderr_str
                }
            )
            
            await self._store_test_result(result)
            
            logger.info(f"Traceroute test {test_id} completed: {status} ({len(hop_details)} hops)")
            return result
            
        except Exception as e:
            logger.error(f"Error in traceroute test {test_id}: {e}")
            
            result = ConnectivityTestResult(
                test_id=test_id,
                source=create_network_endpoint("127.0.0.1"),
                destination=destination,
                test_type=TestType.TRACEROUTE,
                status=TestStatus.FAILURE,
                error_message=str(e),
                timestamp=datetime.now(),
                duration_ms=int((time.time() - start_time) * 1000)
            )
            
            await self._store_test_result(result)
            return result
    
    def _parse_traceroute_output(self, output: str) -> List[NetworkHop]:
        """Parse traceroute output to extract hop information"""
        hops = []
        
        try:
            lines = output.split('\n')
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                # Skip header lines
                if "traceroute" in line.lower() or "tracing route" in line.lower():
                    continue
                if "over a maximum of" in line.lower():
                    continue
                
                if self.is_windows:
                    # Windows tracert output parsing
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        hop_number = int(parts[0])
                        ip_address = None
                        hostname = None
                        latency_ms = None
                        timeout = False
                        
                        # Look for IP address in brackets or at end
                        for part in parts[1:]:
                            if "[" in part and "]" in part:
                                ip_address = part.strip("[]")
                            elif part.replace(".", "").replace(":", "").isdigit():
                                # Potential IP address
                                if part.count(".") == 3:
                                    ip_address = part
                            elif "ms" in part:
                                try:
                                    latency_ms = float(part.replace("ms", ""))
                                except ValueError:
                                    pass
                            elif "*" in part:
                                timeout = True
                        
                        if not ip_address and len(parts) > 1:
                            # Use hostname if no IP found
                            hostname = parts[1]
                            ip_address = hostname  # Fallback
                        
                        if ip_address:
                            hop = NetworkHop(
                                hop_number=hop_number,
                                ip_address=ip_address,
                                hostname=hostname,
                                latency_ms=latency_ms,
                                timeout=timeout
                            )
                            hops.append(hop)
                
                else:
                    # Unix/Linux traceroute output parsing
                    parts = line.split()
                    if len(parts) >= 2 and parts[0].isdigit():
                        hop_number = int(parts[0])
                        
                        # Parse hostname/IP and latency
                        ip_address = None
                        hostname = None
                        latencies = []
                        timeout = False
                        
                        i = 1
                        while i < len(parts):
                            part = parts[i]
                            
                            if "(" in part and ")" in part:
                                # IP address in parentheses
                                ip_address = part.strip("()")
                            elif part == "*":
                                timeout = True
                            elif "ms" in part:
                                try:
                                    latency = float(part.replace("ms", ""))
                                    latencies.append(latency)
                                except ValueError:
                                    pass
                            elif not ip_address and not hostname:
                                # First non-numeric part is likely hostname
                                hostname = part
                                if not ip_address:
                                    ip_address = part  # Use hostname as IP if no IP found
                            
                            i += 1
                        
                        if ip_address:
                            avg_latency = sum(latencies) / len(latencies) if latencies else None
                            
                            hop = NetworkHop(
                                hop_number=hop_number,
                                ip_address=ip_address,
                                hostname=hostname,
                                latency_ms=avg_latency,
                                timeout=timeout
                            )
                            hops.append(hop)
            
        except Exception as e:
            logger.warning(f"Error parsing traceroute output: {e}")
        
        return hops
    
    async def comprehensive_test(
        self,
        destination: NetworkEndpoint,
        include_traceroute: bool = True
    ) -> List[ConnectivityTestResult]:
        """
        Run comprehensive connectivity tests (ping + port + optional traceroute)
        
        Args:
            destination: Target endpoint
            include_traceroute: Whether to include traceroute test
            
        Returns:
            List of test results
        """
        results = []
        
        try:
            logger.info(f"Starting comprehensive connectivity test to {destination.ip_address}")
            
            # Always run ping test
            ping_result = await self.ping_test(destination)
            results.append(ping_result)
            
            # Run port test if port is specified
            if destination.port:
                port_result = await self.port_connectivity_test(destination)
                results.append(port_result)
            
            # Run traceroute if requested and ping was successful
            if include_traceroute and ping_result.status == TestStatus.SUCCESS:
                traceroute_result = await self.traceroute_test(destination)
                results.append(traceroute_result)
            
            logger.info(f"Comprehensive test completed with {len(results)} test results")
            return results
            
        except Exception as e:
            logger.error(f"Error in comprehensive connectivity test: {e}")
            return results
    
    async def _store_test_result(self, result: ConnectivityTestResult):
        """Store test result in database"""
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                cursor.execute("""
                    INSERT INTO connectivity_tests (
                        test_id, source_info, destination_info, test_type, status,
                        latency_ms, packet_loss_percent, hop_details, error_message,
                        timestamp, duration_ms, additional_info
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result.test_id,
                    json.dumps(result.source.to_dict()),
                    json.dumps(result.destination.to_dict()),
                    result.test_type.value,
                    result.status.value,
                    result.latency_ms,
                    result.packet_loss_percent,
                    json.dumps([hop.to_dict() for hop in result.hop_details]),
                    result.error_message,
                    result.timestamp.isoformat(),
                    result.duration_ms,
                    json.dumps(result.additional_info)
                ))
                
                conn.commit()
                logger.debug(f"Stored test result: {result.test_id}")
                
        except Exception as e:
            logger.error(f"Error storing test result: {e}")
    
    async def get_test_history(
        self,
        destination_ip: Optional[str] = None,
        test_type: Optional[TestType] = None,
        limit: int = 100
    ) -> List[ConnectivityTestResult]:
        """
        Get test history from database
        
        Args:
            destination_ip: Filter by destination IP (optional)
            test_type: Filter by test type (optional)
            limit: Maximum number of results
            
        Returns:
            List of historical test results
        """
        try:
            with sqlite3.connect(self.database_path) as conn:
                cursor = conn.cursor()
                
                query = "SELECT * FROM connectivity_tests"
                params = []
                conditions = []
                
                if destination_ip:
                    conditions.append("destination_info LIKE ?")
                    params.append(f'%"{destination_ip}"%')
                
                if test_type:
                    conditions.append("test_type = ?")
                    params.append(test_type.value)
                
                if conditions:
                    query += " WHERE " + " AND ".join(conditions)
                
                query += " ORDER BY timestamp DESC LIMIT ?"
                params.append(limit)
                
                cursor.execute(query, params)
                rows = cursor.fetchall()
                
                results = []
                for row in rows:
                    # Reconstruct ConnectivityTestResult from database row
                    source_data = json.loads(row[1])
                    dest_data = json.loads(row[2])
                    hop_data = json.loads(row[7]) if row[7] else []
                    additional_data = json.loads(row[11]) if row[11] else {}
                    
                    source = NetworkEndpoint(**source_data)
                    destination = NetworkEndpoint(**dest_data)
                    hop_details = [NetworkHop(**hop) for hop in hop_data]
                    
                    result = ConnectivityTestResult(
                        test_id=row[0],
                        source=source,
                        destination=destination,
                        test_type=TestType(row[3]),
                        status=TestStatus(row[4]),
                        latency_ms=row[5],
                        packet_loss_percent=row[6],
                        hop_details=hop_details,
                        error_message=row[8],
                        timestamp=datetime.fromisoformat(row[9]),
                        duration_ms=row[10],
                        additional_info=additional_data
                    )
                    results.append(result)
                
                logger.info(f"Retrieved {len(results)} test results from history")
                return results
                
        except Exception as e:
            logger.error(f"Error retrieving test history: {e}")
            return []
    
    async def get_test_status(self, test_id: str) -> Optional[ConnectivityTestResult]:
        """Get specific test result by ID"""
        try:
            history = await self.get_test_history(limit=1000)  # Get larger set to search
            for result in history:
                if result.test_id == test_id:
                    return result
            return None
            
        except Exception as e:
            logger.error(f"Error getting test status for {test_id}: {e}")
            return None


# Export the main class
__all__ = ["ConnectivityTester"]