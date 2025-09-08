"""
Performance Monitoring System for GCP Security Agent
Comprehensive monitoring with Prometheus metrics, alerting, and dashboard configuration
"""

import time
import asyncio
import threading
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
import psutil
import statistics
from functools import wraps
import weakref
from contextlib import asynccontextmanager
import os
import socket

try:
    from prometheus_client import Counter, Histogram, Gauge, Summary, CollectorRegistry, generate_latest
    from prometheus_client.parser import text_string_to_metric_families
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False
    Counter = Histogram = Gauge = Summary = CollectorRegistry = None

logger = logging.getLogger(__name__)

@dataclass
class AlertRule:
    """Performance alert rule configuration"""
    name: str
    metric_name: str
    condition: str  # 'gt', 'lt', 'eq', 'ne'
    threshold: float
    duration: int = 60  # seconds
    severity: str = 'warning'  # 'info', 'warning', 'error', 'critical'
    description: str = ""
    enabled: bool = True

@dataclass
class PerformanceThresholds:
    """Performance threshold configuration"""
    # Response time thresholds (seconds)
    response_time_warning: float = 1.0
    response_time_critical: float = 2.0
    
    # Memory usage thresholds (percentage)
    memory_warning: float = 80.0
    memory_critical: float = 90.0
    
    # CPU usage thresholds (percentage)
    cpu_warning: float = 80.0
    cpu_critical: float = 90.0
    
    # Database query thresholds (seconds)
    db_query_warning: float = 0.5
    db_query_critical: float = 1.0
    
    # Cache hit rate thresholds (percentage)
    cache_hit_rate_warning: float = 80.0
    cache_hit_rate_critical: float = 60.0
    
    # Request rate thresholds (requests/second)
    request_rate_warning: float = 100.0
    request_rate_critical: float = 500.0
    
    # Error rate thresholds (percentage)
    error_rate_warning: float = 1.0
    error_rate_critical: float = 5.0

@dataclass
class MetricSnapshot:
    """Point-in-time metric snapshot"""
    timestamp: float
    value: float
    labels: Dict[str, str] = field(default_factory=dict)

class TimeSeries:
    """Time series data storage with efficient querying"""
    
    def __init__(self, max_points: int = 10000, retention_seconds: int = 86400):
        self.max_points = max_points
        self.retention_seconds = retention_seconds
        self._data = deque(maxlen=max_points)
        self._lock = threading.RLock()
    
    def add_point(self, timestamp: float, value: float, labels: Dict[str, str] = None):
        """Add a data point to the time series"""
        with self._lock:
            # Clean old data
            self._cleanup_old_data(timestamp)
            
            point = MetricSnapshot(timestamp, value, labels or {})
            self._data.append(point)
    
    def _cleanup_old_data(self, current_time: float):
        """Remove data points older than retention period"""
        cutoff_time = current_time - self.retention_seconds
        
        # Remove old points from the left
        while self._data and self._data[0].timestamp < cutoff_time:
            self._data.popleft()
    
    def get_range(self, start_time: float, end_time: float) -> List[MetricSnapshot]:
        """Get data points within time range"""
        with self._lock:
            return [
                point for point in self._data
                if start_time <= point.timestamp <= end_time
            ]
    
    def get_latest(self, count: int = 1) -> List[MetricSnapshot]:
        """Get the latest N data points"""
        with self._lock:
            return list(self._data)[-count:] if self._data else []
    
    def aggregate(self, start_time: float, end_time: float, 
                 aggregation: str = 'avg') -> Optional[float]:
        """Aggregate data points in time range"""
        points = self.get_range(start_time, end_time)
        if not points:
            return None
        
        values = [point.value for point in points]
        
        if aggregation == 'avg':
            return statistics.mean(values)
        elif aggregation == 'sum':
            return sum(values)
        elif aggregation == 'min':
            return min(values)
        elif aggregation == 'max':
            return max(values)
        elif aggregation == 'median':
            return statistics.median(values)
        elif aggregation == 'p95':
            return statistics.quantiles(values, n=20)[18] if len(values) > 1 else values[0]
        elif aggregation == 'p99':
            return statistics.quantiles(values, n=100)[98] if len(values) > 1 else values[0]
        else:
            return statistics.mean(values)
    
    def size(self) -> int:
        """Get number of data points"""
        with self._lock:
            return len(self._data)

class PrometheusMetrics:
    """Prometheus metrics collector"""
    
    def __init__(self, registry: CollectorRegistry = None):
        if not PROMETHEUS_AVAILABLE:
            logger.warning("Prometheus client not available")
            return
        
        self.registry = registry or CollectorRegistry()
        self._setup_metrics()
    
    def _setup_metrics(self):
        """Initialize Prometheus metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # HTTP request metrics
        self.http_requests_total = Counter(
            'http_requests_total',
            'Total HTTP requests',
            ['method', 'endpoint', 'status'],
            registry=self.registry
        )
        
        self.http_request_duration = Histogram(
            'http_request_duration_seconds',
            'HTTP request duration',
            ['method', 'endpoint'],
            buckets=[0.1, 0.5, 1.0, 2.0, 5.0, 10.0],
            registry=self.registry
        )
        
        # Database metrics
        self.db_queries_total = Counter(
            'db_queries_total',
            'Total database queries',
            ['operation', 'table'],
            registry=self.registry
        )
        
        self.db_query_duration = Histogram(
            'db_query_duration_seconds',
            'Database query duration',
            ['operation', 'table'],
            buckets=[0.01, 0.05, 0.1, 0.5, 1.0, 2.0],
            registry=self.registry
        )
        
        # Cache metrics
        self.cache_operations_total = Counter(
            'cache_operations_total',
            'Total cache operations',
            ['operation', 'tier'],
            registry=self.registry
        )
        
        self.cache_hit_rate = Gauge(
            'cache_hit_rate',
            'Cache hit rate percentage',
            ['tier'],
            registry=self.registry
        )
        
        # System metrics
        self.system_cpu_usage = Gauge(
            'system_cpu_usage_percent',
            'System CPU usage percentage',
            registry=self.registry
        )
        
        self.system_memory_usage = Gauge(
            'system_memory_usage_percent',
            'System memory usage percentage',
            registry=self.registry
        )
        
        self.system_disk_usage = Gauge(
            'system_disk_usage_percent',
            'System disk usage percentage',
            ['device'],
            registry=self.registry
        )
        
        # Application metrics
        self.active_connections = Gauge(
            'active_connections',
            'Number of active connections',
            registry=self.registry
        )
        
        self.request_queue_size = Gauge(
            'request_queue_size',
            'Request queue size',
            registry=self.registry
        )
        
        self.error_rate = Gauge(
            'error_rate_percent',
            'Error rate percentage',
            ['service'],
            registry=self.registry
        )
    
    def record_http_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.http_requests_total.labels(
            method=method, endpoint=endpoint, status=str(status)
        ).inc()
        
        self.http_request_duration.labels(
            method=method, endpoint=endpoint
        ).observe(duration)
    
    def record_db_query(self, operation: str, table: str, duration: float):
        """Record database query metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.db_queries_total.labels(operation=operation, table=table).inc()
        self.db_query_duration.labels(operation=operation, table=table).observe(duration)
    
    def record_cache_operation(self, operation: str, tier: str, hit: bool = None):
        """Record cache operation metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.cache_operations_total.labels(operation=operation, tier=tier).inc()
    
    def update_cache_hit_rate(self, tier: str, hit_rate: float):
        """Update cache hit rate"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        self.cache_hit_rate.labels(tier=tier).set(hit_rate * 100)
    
    def update_system_metrics(self):
        """Update system resource metrics"""
        if not PROMETHEUS_AVAILABLE:
            return
        
        # CPU usage
        cpu_percent = psutil.cpu_percent(interval=1)
        self.system_cpu_usage.set(cpu_percent)
        
        # Memory usage
        memory = psutil.virtual_memory()
        self.system_memory_usage.set(memory.percent)
        
        # Disk usage
        for partition in psutil.disk_partitions():
            try:
                usage = psutil.disk_usage(partition.mountpoint)
                self.system_disk_usage.labels(device=partition.device).set(
                    (usage.used / usage.total) * 100
                )
            except PermissionError:
                pass
    
    def get_metrics(self) -> str:
        """Get metrics in Prometheus format"""
        if not PROMETHEUS_AVAILABLE:
            return ""
        
        return generate_latest(self.registry).decode('utf-8')

class AlertManager:
    """Alert manager for performance monitoring"""
    
    def __init__(self, thresholds: PerformanceThresholds = None):
        self.thresholds = thresholds or PerformanceThresholds()
        self.alert_rules = []
        self.active_alerts = {}
        self.alert_callbacks = []
        self._lock = threading.RLock()
        
        self._setup_default_rules()
    
    def _setup_default_rules(self):
        """Setup default alert rules"""
        default_rules = [
            AlertRule(
                name="high_response_time",
                metric_name="http_request_duration",
                condition="gt",
                threshold=self.thresholds.response_time_warning,
                severity="warning",
                description="HTTP response time is high"
            ),
            AlertRule(
                name="critical_response_time",
                metric_name="http_request_duration",
                condition="gt",
                threshold=self.thresholds.response_time_critical,
                severity="critical",
                description="HTTP response time is critically high"
            ),
            AlertRule(
                name="high_memory_usage",
                metric_name="system_memory_usage",
                condition="gt",
                threshold=self.thresholds.memory_warning,
                severity="warning",
                description="Memory usage is high"
            ),
            AlertRule(
                name="high_error_rate",
                metric_name="error_rate",
                condition="gt",
                threshold=self.thresholds.error_rate_warning,
                severity="warning",
                description="Error rate is high"
            ),
            AlertRule(
                name="low_cache_hit_rate",
                metric_name="cache_hit_rate",
                condition="lt",
                threshold=self.thresholds.cache_hit_rate_warning,
                severity="warning",
                description="Cache hit rate is low"
            )
        ]
        
        self.alert_rules.extend(default_rules)
    
    def add_alert_rule(self, rule: AlertRule):
        """Add custom alert rule"""
        with self._lock:
            self.alert_rules.append(rule)
    
    def add_alert_callback(self, callback: Callable[[str, AlertRule, float], None]):
        """Add callback for alert notifications"""
        self.alert_callbacks.append(callback)
    
    def evaluate_alerts(self, metrics: Dict[str, float]):
        """Evaluate alert rules against current metrics"""
        current_time = time.time()
        
        with self._lock:
            for rule in self.alert_rules:
                if not rule.enabled or rule.metric_name not in metrics:
                    continue
                
                metric_value = metrics[rule.metric_name]
                alert_key = f"{rule.name}_{rule.metric_name}"
                
                # Check if condition is met
                condition_met = self._evaluate_condition(
                    metric_value, rule.condition, rule.threshold
                )
                
                if condition_met:
                    if alert_key not in self.active_alerts:
                        # New alert
                        self.active_alerts[alert_key] = {
                            'rule': rule,
                            'start_time': current_time,
                            'last_seen': current_time,
                            'value': metric_value
                        }
                        
                        # Check if alert duration threshold is met
                        if current_time - self.active_alerts[alert_key]['start_time'] >= rule.duration:
                            self._trigger_alert(rule, metric_value)
                    else:
                        # Update existing alert
                        self.active_alerts[alert_key]['last_seen'] = current_time
                        self.active_alerts[alert_key]['value'] = metric_value
                
                else:
                    # Condition not met, resolve alert if active
                    if alert_key in self.active_alerts:
                        self._resolve_alert(alert_key)
    
    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition"""
        if condition == 'gt':
            return value > threshold
        elif condition == 'lt':
            return value < threshold
        elif condition == 'eq':
            return abs(value - threshold) < 0.001  # Float comparison
        elif condition == 'ne':
            return abs(value - threshold) >= 0.001
        return False
    
    def _trigger_alert(self, rule: AlertRule, value: float):
        """Trigger alert notification"""
        logger.warning(f"ALERT: {rule.name} - {rule.description} (value: {value})")
        
        # Call alert callbacks
        for callback in self.alert_callbacks:
            try:
                callback(rule.name, rule, value)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")
    
    def _resolve_alert(self, alert_key: str):
        """Resolve active alert"""
        if alert_key in self.active_alerts:
            alert = self.active_alerts[alert_key]
            logger.info(f"RESOLVED: {alert['rule'].name}")
            del self.active_alerts[alert_key]
    
    def get_active_alerts(self) -> List[Dict[str, Any]]:
        """Get list of active alerts"""
        with self._lock:
            return [
                {
                    'name': alert['rule'].name,
                    'severity': alert['rule'].severity,
                    'description': alert['rule'].description,
                    'value': alert['value'],
                    'duration': time.time() - alert['start_time']
                }
                for alert in self.active_alerts.values()
            ]

class PerformanceMonitor:
    """Main performance monitoring system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.thresholds = PerformanceThresholds()
        
        # Initialize components
        self.metrics = {}
        self.time_series = defaultdict(lambda: TimeSeries())
        self.prometheus = PrometheusMetrics() if PROMETHEUS_AVAILABLE else None
        self.alert_manager = AlertManager(self.thresholds)
        
        # Monitoring state
        self._monitoring = False
        self._monitor_task = None
        self._lock = threading.RLock()
        
        # Performance counters
        self.request_counters = defaultdict(int)
        self.error_counters = defaultdict(int)
        self.timing_data = defaultdict(list)
        
        # Start system monitoring
        self._start_system_monitoring()
    
    def _start_system_monitoring(self):
        """Start background system monitoring"""
        def monitor_system():
            while self._monitoring:
                try:
                    current_time = time.time()
                    
                    # Collect system metrics
                    cpu_percent = psutil.cpu_percent(interval=1)
                    memory = psutil.virtual_memory()
                    
                    # Store in time series
                    self.record_metric('system_cpu_usage', cpu_percent, current_time)
                    self.record_metric('system_memory_usage', memory.percent, current_time)
                    
                    # Update Prometheus metrics
                    if self.prometheus:
                        self.prometheus.update_system_metrics()
                    
                    # Evaluate alerts
                    current_metrics = {
                        'system_cpu_usage': cpu_percent,
                        'system_memory_usage': memory.percent
                    }
                    self.alert_manager.evaluate_alerts(current_metrics)
                    
                    time.sleep(10)  # Monitor every 10 seconds
                
                except Exception as e:
                    logger.error(f"System monitoring error: {e}")
        
        self._monitoring = True
        self._monitor_task = threading.Thread(target=monitor_system, daemon=True)
        self._monitor_task.start()
    
    def record_metric(self, name: str, value: float, timestamp: float = None, 
                     labels: Dict[str, str] = None):
        """Record a metric value"""
        if timestamp is None:
            timestamp = time.time()
        
        with self._lock:
            # Store in time series
            self.time_series[name].add_point(timestamp, value, labels)
            
            # Store latest value
            self.metrics[name] = value
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record HTTP request metrics"""
        current_time = time.time()
        
        # Update counters
        self.request_counters[f"{method}_{endpoint}"] += 1
        if status >= 400:
            self.error_counters[f"{method}_{endpoint}"] += 1
        
        # Record duration
        self.timing_data[f"{method}_{endpoint}"].append(duration)
        
        # Store metrics
        self.record_metric(f"http_request_duration_{method}_{endpoint}", duration, current_time)
        self.record_metric("http_requests_total", self.request_counters[f"{method}_{endpoint}"], current_time)
        
        # Update Prometheus
        if self.prometheus:
            self.prometheus.record_http_request(method, endpoint, status, duration)
        
        # Check for slow requests
        if duration > self.thresholds.response_time_warning:
            logger.warning(f"Slow request: {method} {endpoint} took {duration:.2f}s")
    
    def record_db_operation(self, operation: str, table: str, duration: float):
        """Record database operation metrics"""
        current_time = time.time()
        
        metric_name = f"db_{operation}_{table}"
        self.record_metric(metric_name, duration, current_time)
        
        if self.prometheus:
            self.prometheus.record_db_query(operation, table, duration)
        
        # Check for slow queries
        if duration > self.thresholds.db_query_warning:
            logger.warning(f"Slow DB query: {operation} on {table} took {duration:.2f}s")
    
    def record_cache_operation(self, operation: str, tier: str, hit: bool):
        """Record cache operation metrics"""
        current_time = time.time()
        
        # Update hit rate
        cache_key = f"cache_{tier}"
        if cache_key not in self.timing_data:
            self.timing_data[cache_key] = {'hits': 0, 'total': 0}
        
        self.timing_data[cache_key]['total'] += 1
        if hit:
            self.timing_data[cache_key]['hits'] += 1
        
        hit_rate = self.timing_data[cache_key]['hits'] / self.timing_data[cache_key]['total']
        self.record_metric(f"cache_hit_rate_{tier}", hit_rate * 100, current_time)
        
        if self.prometheus:
            self.prometheus.record_cache_operation(operation, tier, hit)
            self.prometheus.update_cache_hit_rate(tier, hit_rate)
        
        # Check hit rate
        if hit_rate * 100 < self.thresholds.cache_hit_rate_warning:
            logger.warning(f"Low cache hit rate for {tier}: {hit_rate*100:.1f}%")
    
    def get_metrics(self, time_range: int = 3600) -> Dict[str, Any]:
        """Get performance metrics for specified time range"""
        current_time = time.time()
        start_time = current_time - time_range
        
        result = {
            'timestamp': current_time,
            'time_range': time_range,
            'metrics': {},
            'alerts': self.alert_manager.get_active_alerts()
        }
        
        # Aggregate time series data
        for metric_name, ts in self.time_series.items():
            try:
                avg_value = ts.aggregate(start_time, current_time, 'avg')
                max_value = ts.aggregate(start_time, current_time, 'max')
                min_value = ts.aggregate(start_time, current_time, 'min')
                p95_value = ts.aggregate(start_time, current_time, 'p95')
                
                result['metrics'][metric_name] = {
                    'avg': avg_value,
                    'max': max_value,
                    'min': min_value,
                    'p95': p95_value,
                    'current': self.metrics.get(metric_name, 0),
                    'data_points': ts.size()
                }
            except Exception as e:
                logger.error(f"Error aggregating metric {metric_name}: {e}")
        
        # Add request statistics
        if self.request_counters:
            total_requests = sum(self.request_counters.values())
            total_errors = sum(self.error_counters.values())
            error_rate = (total_errors / total_requests * 100) if total_requests > 0 else 0
            
            result['request_stats'] = {
                'total_requests': total_requests,
                'total_errors': total_errors,
                'error_rate': error_rate,
                'requests_per_second': total_requests / time_range
            }
        
        return result
    
    def get_dashboard_config(self) -> Dict[str, Any]:
        """Generate dashboard configuration for monitoring tools"""
        return {
            'dashboard': {
                'title': 'GCP Security Agent Performance',
                'refresh': '30s',
                'panels': [
                    {
                        'title': 'Response Time',
                        'type': 'graph',
                        'metrics': ['http_request_duration'],
                        'thresholds': [
                            {'value': self.thresholds.response_time_warning, 'color': 'yellow'},
                            {'value': self.thresholds.response_time_critical, 'color': 'red'}
                        ]
                    },
                    {
                        'title': 'System Resources',
                        'type': 'graph',
                        'metrics': ['system_cpu_usage', 'system_memory_usage'],
                        'thresholds': [
                            {'value': self.thresholds.cpu_warning, 'color': 'yellow'},
                            {'value': self.thresholds.memory_warning, 'color': 'yellow'}
                        ]
                    },
                    {
                        'title': 'Cache Hit Rates',
                        'type': 'graph',
                        'metrics': ['cache_hit_rate_memory', 'cache_hit_rate_redis', 'cache_hit_rate_sqlite'],
                        'thresholds': [
                            {'value': self.thresholds.cache_hit_rate_warning, 'color': 'yellow'}
                        ]
                    },
                    {
                        'title': 'Database Performance',
                        'type': 'graph',
                        'metrics': ['db_query_duration'],
                        'thresholds': [
                            {'value': self.thresholds.db_query_warning, 'color': 'yellow'},
                            {'value': self.thresholds.db_query_critical, 'color': 'red'}
                        ]
                    },
                    {
                        'title': 'Request Rate',
                        'type': 'stat',
                        'metrics': ['http_requests_total'],
                        'unit': 'reqps'
                    },
                    {
                        'title': 'Error Rate',
                        'type': 'stat',
                        'metrics': ['error_rate'],
                        'unit': 'percent',
                        'thresholds': [
                            {'value': self.thresholds.error_rate_warning, 'color': 'yellow'},
                            {'value': self.thresholds.error_rate_critical, 'color': 'red'}
                        ]
                    }
                ]
            },
            'alerts': [rule.__dict__ for rule in self.alert_manager.alert_rules]
        }
    
    def export_prometheus_metrics(self) -> str:
        """Export metrics in Prometheus format"""
        if self.prometheus:
            return self.prometheus.get_metrics()
        return ""
    
    def add_alert_callback(self, callback: Callable):
        """Add alert notification callback"""
        self.alert_manager.add_alert_callback(callback)
    
    def stop(self):
        """Stop performance monitoring"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.join(timeout=5)

# Performance monitoring decorators
def monitor_performance(metric_name: str = None, record_args: bool = False):
    """Decorator to monitor function performance"""
    def decorator(func):
        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metric
                monitor.record_metric(f"{name}_duration", duration)
                monitor.record_metric(f"{name}_success", 1)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_metric(f"{name}_duration", duration)
                monitor.record_metric(f"{name}_error", 1)
                raise
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs):
            name = metric_name or f"{func.__module__}.{func.__name__}"
            start_time = time.time()
            
            try:
                result = await func(*args, **kwargs)
                duration = time.time() - start_time
                
                # Record metric
                monitor.record_metric(f"{name}_duration", duration)
                monitor.record_metric(f"{name}_success", 1)
                
                return result
            except Exception as e:
                duration = time.time() - start_time
                monitor.record_metric(f"{name}_duration", duration)
                monitor.record_metric(f"{name}_error", 1)
                raise
        
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        else:
            return sync_wrapper
    
    return decorator

# Alert notification functions
def slack_alert_callback(webhook_url: str):
    """Create Slack alert callback"""
    def callback(alert_name: str, rule: AlertRule, value: float):
        try:
            import requests
            
            message = {
                "text": f"🚨 ALERT: {rule.name}",
                "attachments": [{
                    "color": "danger" if rule.severity == "critical" else "warning",
                    "fields": [
                        {"title": "Description", "value": rule.description, "short": False},
                        {"title": "Metric", "value": rule.metric_name, "short": True},
                        {"title": "Value", "value": f"{value:.2f}", "short": True},
                        {"title": "Threshold", "value": f"{rule.threshold}", "short": True},
                        {"title": "Severity", "value": rule.severity, "short": True}
                    ]
                }]
            }
            
            requests.post(webhook_url, json=message, timeout=10)
        except Exception as e:
            logger.error(f"Slack alert callback failed: {e}")
    
    return callback

def email_alert_callback(smtp_config: Dict[str, Any]):
    """Create email alert callback"""
    def callback(alert_name: str, rule: AlertRule, value: float):
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = smtp_config['from']
            msg['To'] = smtp_config['to']
            msg['Subject'] = f"Performance Alert: {rule.name}"
            
            body = f"""
            Performance Alert Triggered
            
            Alert: {rule.name}
            Description: {rule.description}
            Metric: {rule.metric_name}
            Current Value: {value:.2f}
            Threshold: {rule.threshold}
            Severity: {rule.severity}
            
            Please investigate immediately.
            """
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(smtp_config['host'], smtp_config['port'])
            if smtp_config.get('use_tls'):
                server.starttls()
            if smtp_config.get('username'):
                server.login(smtp_config['username'], smtp_config['password'])
            
            server.send_message(msg)
            server.quit()
            
        except Exception as e:
            logger.error(f"Email alert callback failed: {e}")
    
    return callback

# Global performance monitor instance
monitor = PerformanceMonitor()

# Context manager for monitoring operations
@asynccontextmanager
async def monitor_operation(operation_name: str):
    """Context manager for monitoring async operations"""
    start_time = time.time()
    try:
        yield
        duration = time.time() - start_time
        monitor.record_metric(f"{operation_name}_duration", duration)
        monitor.record_metric(f"{operation_name}_success", 1)
    except Exception as e:
        duration = time.time() - start_time
        monitor.record_metric(f"{operation_name}_duration", duration)
        monitor.record_metric(f"{operation_name}_error", 1)
        raise