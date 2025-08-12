"""Performance monitoring and real-time metrics API for chat-centric architecture."""

import asyncio
import time
import psutil
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel
from dataclasses import dataclass, field
import uuid

logger = logging.getLogger(__name__)
router = APIRouter()

@dataclass
class PerformanceMetric:
    """Represents a performance metric with metadata."""
    metric_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=datetime.now)
    metric_type: str = "general"
    value: float = 0.0
    unit: str = ""
    source: str = "system"
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)

class SystemMetrics(BaseModel):
    """System performance metrics model."""
    cpu_percent: float
    memory_percent: float
    disk_usage_percent: float
    network_io: Dict[str, int]
    process_count: int
    load_average: List[float]
    timestamp: str

class ResponseTimeMetric(BaseModel):
    """Response time metrics model."""
    endpoint: str
    method: str
    response_time_ms: float
    status_code: int
    user_id: Optional[str] = None
    agent_used: Optional[str] = None
    timestamp: str

class AgentPerformanceMetric(BaseModel):
    """Agent-specific performance metrics."""
    agent_name: str
    delegation_time_ms: float
    processing_time_ms: float
    context_analysis_time_ms: float
    response_generation_time_ms: float
    success_rate: float
    error_count: int
    timestamp: str

class PerformanceMonitor:
    """Real-time performance monitoring system."""
    
    def __init__(self):
        self.metrics_history: Dict[str, List[PerformanceMetric]] = {}
        self.system_metrics: List[SystemMetrics] = []
        self.response_times: List[ResponseTimeMetric] = []
        self.agent_metrics: List[AgentPerformanceMetric] = []
        self.alert_thresholds = {
            "cpu_percent": 80.0,
            "memory_percent": 85.0,
            "disk_usage_percent": 90.0,
            "response_time_ms": 5000.0
        }
        self.monitoring_active = False
        self._monitor_task = None
        
    async def start_monitoring(self):
        """Start background performance monitoring."""
        if not self.monitoring_active:
            self.monitoring_active = True
            self._monitor_task = asyncio.create_task(self._monitor_system_metrics())
            logger.info("Performance monitoring started")
    
    async def stop_monitoring(self):
        """Stop background performance monitoring."""
        self.monitoring_active = False
        if self._monitor_task:
            self._monitor_task.cancel()
            logger.info("Performance monitoring stopped")
    
    async def _monitor_system_metrics(self):
        """Background task to collect system metrics."""
        while self.monitoring_active:
            try:
                # Collect system metrics
                cpu_percent = psutil.cpu_percent(interval=1)
                memory = psutil.virtual_memory()
                disk = psutil.disk_usage('/')
                network = psutil.net_io_counters()
                
                metric = SystemMetrics(
                    cpu_percent=cpu_percent,
                    memory_percent=memory.percent,
                    disk_usage_percent=(disk.used / disk.total) * 100,
                    network_io={
                        "bytes_sent": network.bytes_sent,
                        "bytes_recv": network.bytes_recv,
                        "packets_sent": network.packets_sent,
                        "packets_recv": network.packets_recv
                    },
                    process_count=len(psutil.pids()),
                    load_average=list(psutil.getloadavg()) if hasattr(psutil, 'getloadavg') else [0.0, 0.0, 0.0],
                    timestamp=datetime.now().isoformat()
                )
                
                self.system_metrics.append(metric)
                
                # Keep only last 1000 entries
                if len(self.system_metrics) > 1000:
                    self.system_metrics = self.system_metrics[-1000:]
                
                # Check for alerts
                await self._check_alerts(metric)
                
                await asyncio.sleep(30)  # Collect every 30 seconds
                
            except Exception as e:
                logger.error(f"Error collecting system metrics: {e}")
                await asyncio.sleep(30)
    
    async def _check_alerts(self, metric: SystemMetrics):
        """Check if any metrics exceed alert thresholds."""
        alerts = []
        
        if metric.cpu_percent > self.alert_thresholds["cpu_percent"]:
            alerts.append(f"High CPU usage: {metric.cpu_percent:.1f}%")
        
        if metric.memory_percent > self.alert_thresholds["memory_percent"]:
            alerts.append(f"High memory usage: {metric.memory_percent:.1f}%")
        
        if metric.disk_usage_percent > self.alert_thresholds["disk_usage_percent"]:
            alerts.append(f"High disk usage: {metric.disk_usage_percent:.1f}%")
        
        if alerts:
            logger.warning(f"Performance alerts: {', '.join(alerts)}")
            # Here you could send alerts via WebSocket or other notification system
    
    def record_response_time(
        self, 
        endpoint: str, 
        method: str, 
        response_time_ms: float,
        status_code: int,
        user_id: str = None,
        agent_used: str = None
    ):
        """Record API response time metric."""
        metric = ResponseTimeMetric(
            endpoint=endpoint,
            method=method,
            response_time_ms=response_time_ms,
            status_code=status_code,
            user_id=user_id,
            agent_used=agent_used,
            timestamp=datetime.now().isoformat()
        )
        
        self.response_times.append(metric)
        
        # Keep only last 10000 entries
        if len(self.response_times) > 10000:
            self.response_times = self.response_times[-10000:]
        
        # Check for slow response alert
        if response_time_ms > self.alert_thresholds["response_time_ms"]:
            logger.warning(f"Slow response detected: {endpoint} took {response_time_ms:.0f}ms")
    
    def record_agent_performance(
        self,
        agent_name: str,
        delegation_time_ms: float,
        processing_time_ms: float,
        context_analysis_time_ms: float,
        response_generation_time_ms: float,
        success: bool
    ):
        """Record agent-specific performance metrics."""
        # Calculate success rate from recent metrics
        recent_metrics = [m for m in self.agent_metrics 
                         if m.agent_name == agent_name and 
                         datetime.fromisoformat(m.timestamp) > datetime.now() - timedelta(hours=1)]
        
        success_count = len([m for m in recent_metrics if m.error_count == 0])
        total_count = len(recent_metrics) + 1
        success_rate = (success_count + (1 if success else 0)) / total_count
        
        metric = AgentPerformanceMetric(
            agent_name=agent_name,
            delegation_time_ms=delegation_time_ms,
            processing_time_ms=processing_time_ms,
            context_analysis_time_ms=context_analysis_time_ms,
            response_generation_time_ms=response_generation_time_ms,
            success_rate=success_rate,
            error_count=0 if success else 1,
            timestamp=datetime.now().isoformat()
        )
        
        self.agent_metrics.append(metric)
        
        # Keep only last 5000 entries
        if len(self.agent_metrics) > 5000:
            self.agent_metrics = self.agent_metrics[-5000:]
    
    def get_current_system_metrics(self) -> Optional[SystemMetrics]:
        """Get the most recent system metrics."""
        return self.system_metrics[-1] if self.system_metrics else None
    
    def get_system_metrics_history(self, hours: int = 1) -> List[SystemMetrics]:
        """Get system metrics history for specified hours."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        return [
            metric for metric in self.system_metrics
            if datetime.fromisoformat(metric.timestamp) > cutoff_time
        ]
    
    def get_response_time_stats(self, hours: int = 1) -> Dict[str, Any]:
        """Get response time statistics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            metric for metric in self.response_times
            if datetime.fromisoformat(metric.timestamp) > cutoff_time
        ]
        
        if not recent_metrics:
            return {}
        
        response_times = [m.response_time_ms for m in recent_metrics]
        
        return {
            "total_requests": len(recent_metrics),
            "avg_response_time_ms": sum(response_times) / len(response_times),
            "min_response_time_ms": min(response_times),
            "max_response_time_ms": max(response_times),
            "p95_response_time_ms": self._percentile(response_times, 0.95),
            "p99_response_time_ms": self._percentile(response_times, 0.99),
            "success_rate": len([m for m in recent_metrics if m.status_code < 400]) / len(recent_metrics),
            "endpoints": list(set(m.endpoint for m in recent_metrics))
        }
    
    def get_agent_performance_stats(self, agent_name: str = None, hours: int = 1) -> Dict[str, Any]:
        """Get agent performance statistics."""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        recent_metrics = [
            metric for metric in self.agent_metrics
            if datetime.fromisoformat(metric.timestamp) > cutoff_time and
               (agent_name is None or metric.agent_name == agent_name)
        ]
        
        if not recent_metrics:
            return {}
        
        # Aggregate by agent
        agent_stats = {}
        for metric in recent_metrics:
            if metric.agent_name not in agent_stats:
                agent_stats[metric.agent_name] = {
                    "delegation_times": [],
                    "processing_times": [],
                    "total_requests": 0,
                    "errors": 0
                }
            
            stats = agent_stats[metric.agent_name]
            stats["delegation_times"].append(metric.delegation_time_ms)
            stats["processing_times"].append(metric.processing_time_ms)
            stats["total_requests"] += 1
            stats["errors"] += metric.error_count
        
        # Calculate statistics
        result = {}
        for agent, stats in agent_stats.items():
            result[agent] = {
                "total_requests": stats["total_requests"],
                "avg_delegation_time_ms": sum(stats["delegation_times"]) / len(stats["delegation_times"]),
                "avg_processing_time_ms": sum(stats["processing_times"]) / len(stats["processing_times"]),
                "success_rate": (stats["total_requests"] - stats["errors"]) / stats["total_requests"],
                "error_rate": stats["errors"] / stats["total_requests"]
            }
        
        return result
    
    def _percentile(self, data: List[float], percentile: float) -> float:
        """Calculate percentile of a list of numbers."""
        if not data:
            return 0.0
        
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        if index >= len(sorted_data):
            index = len(sorted_data) - 1
        
        return sorted_data[index]

# Global performance monitor instance
performance_monitor = PerformanceMonitor()

# API Endpoints
@router.get("/system")
async def get_system_metrics(hours: int = 1):
    """Get current and historical system metrics."""
    try:
        current = performance_monitor.get_current_system_metrics()
        history = performance_monitor.get_system_metrics_history(hours)
        
        return {
            "success": True,
            "current": current,
            "history": history,
            "monitoring_active": performance_monitor.monitoring_active
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get system metrics: {str(e)}")

@router.get("/response-times")
async def get_response_time_metrics(hours: int = 1):
    """Get response time statistics."""
    try:
        stats = performance_monitor.get_response_time_stats(hours)
        
        return {
            "success": True,
            "stats": stats,
            "timeframe_hours": hours
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get response time metrics: {str(e)}")

@router.get("/agents")
async def get_agent_performance_metrics(agent_name: str = None, hours: int = 1):
    """Get agent performance statistics."""
    try:
        stats = performance_monitor.get_agent_performance_stats(agent_name, hours)
        
        return {
            "success": True,
            "agent_name": agent_name,
            "stats": stats,
            "timeframe_hours": hours
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get agent performance metrics: {str(e)}")

@router.get("/dashboard")
async def get_performance_dashboard():
    """Get comprehensive performance dashboard data."""
    try:
        current_system = performance_monitor.get_current_system_metrics()
        response_stats = performance_monitor.get_response_time_stats(1)
        agent_stats = performance_monitor.get_agent_performance_stats(hours=1)
        
        # Calculate health score
        health_score = 100.0
        if current_system:
            if current_system.cpu_percent > 80:
                health_score -= 20
            if current_system.memory_percent > 85:
                health_score -= 20
            if current_system.disk_usage_percent > 90:
                health_score -= 20
        
        if response_stats and response_stats.get("avg_response_time_ms", 0) > 2000:
            health_score -= 15
        
        return {
            "success": True,
            "health_score": max(0, health_score),
            "system_metrics": current_system,
            "response_time_stats": response_stats,
            "agent_performance": agent_stats,
            "monitoring_active": performance_monitor.monitoring_active,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get performance dashboard: {str(e)}")

@router.post("/start-monitoring")
async def start_monitoring(background_tasks: BackgroundTasks):
    """Start performance monitoring."""
    try:
        if not performance_monitor.monitoring_active:
            background_tasks.add_task(performance_monitor.start_monitoring)
            
        return {
            "success": True,
            "message": "Performance monitoring started",
            "monitoring_active": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to start monitoring: {str(e)}")

@router.post("/stop-monitoring")
async def stop_monitoring(background_tasks: BackgroundTasks):
    """Stop performance monitoring."""
    try:
        if performance_monitor.monitoring_active:
            background_tasks.add_task(performance_monitor.stop_monitoring)
            
        return {
            "success": True,
            "message": "Performance monitoring stopped",
            "monitoring_active": False
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop monitoring: {str(e)}")

@router.get("/alerts")
async def get_performance_alerts():
    """Get current performance alerts and thresholds."""
    try:
        current_system = performance_monitor.get_current_system_metrics()
        alerts = []
        
        if current_system:
            if current_system.cpu_percent > performance_monitor.alert_thresholds["cpu_percent"]:
                alerts.append({
                    "type": "cpu_high",
                    "message": f"High CPU usage: {current_system.cpu_percent:.1f}%",
                    "severity": "warning",
                    "value": current_system.cpu_percent,
                    "threshold": performance_monitor.alert_thresholds["cpu_percent"]
                })
            
            if current_system.memory_percent > performance_monitor.alert_thresholds["memory_percent"]:
                alerts.append({
                    "type": "memory_high",
                    "message": f"High memory usage: {current_system.memory_percent:.1f}%",
                    "severity": "warning",
                    "value": current_system.memory_percent,
                    "threshold": performance_monitor.alert_thresholds["memory_percent"]
                })
        
        return {
            "success": True,
            "alerts": alerts,
            "thresholds": performance_monitor.alert_thresholds,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get alerts: {str(e)}")