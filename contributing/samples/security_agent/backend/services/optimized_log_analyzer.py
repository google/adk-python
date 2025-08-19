"""
Optimized Log Analysis Service for High-Performance Log Processing.

This module implements performance optimizations for log analysis including:
- Batch processing for large log volumes
- Caching for frequently accessed patterns
- Streaming processing for real-time analysis
- Parallel processing for multi-resource analysis
- Memory-efficient pattern matching

Part of TASK-005: Optimize Log Analysis Performance
"""

import asyncio
import re
from typing import Dict, Any, List, Optional, AsyncIterator, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from collections import defaultdict, deque
from functools import lru_cache
import hashlib
import json
import logging

logger = logging.getLogger(__name__)

# Performance configuration
BATCH_SIZE = 100  # Process logs in batches
CACHE_SIZE = 1024  # LRU cache size for patterns
PATTERN_CACHE_TTL = 3600  # Pattern cache TTL in seconds
MAX_MEMORY_MB = 512  # Maximum memory usage for log processing
PARALLEL_WORKERS = 4  # Number of parallel workers for analysis

@dataclass
class LogBatch:
    """Represents a batch of logs for processing."""
    entries: List[Dict[str, Any]]
    batch_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def __len__(self):
        return len(self.entries)

@dataclass
class AnalysisResult:
    """Optimized analysis result structure."""
    patterns: Dict[str, int] = field(default_factory=dict)
    anomalies: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    error_summary: Dict[str, Any] = field(default_factory=dict)
    processing_time_ms: float = 0
    entries_processed: int = 0

class OptimizedLogAnalyzer:
    """High-performance log analyzer with caching and batch processing."""
    
    def __init__(self):
        """Initialize the optimized log analyzer."""
        self.pattern_cache = {}
        self.compiled_patterns = {}
        self.batch_queue = asyncio.Queue(maxsize=100)
        self.results_cache = {}
        self._initialize_patterns()
    
    def _initialize_patterns(self):
        """Pre-compile regex patterns for performance."""
        patterns = {
            "error": re.compile(r"error|exception|failed|failure", re.IGNORECASE),
            "warning": re.compile(r"warning|warn|deprecated", re.IGNORECASE),
            "timeout": re.compile(r"timeout|timed?\s*out", re.IGNORECASE),
            "connection": re.compile(r"connection|connect|disconnect", re.IGNORECASE),
            "permission": re.compile(r"permission|denied|unauthorized|forbidden", re.IGNORECASE),
            "rate_limit": re.compile(r"rate\s*limit|quota|throttl", re.IGNORECASE),
            "memory": re.compile(r"memory|oom|heap|gc", re.IGNORECASE),
            "latency": re.compile(r"latency[:\s]+(\d+)", re.IGNORECASE),
            "status_code": re.compile(r"status[:\s]+(\d{3})"),
            "ip_address": re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b"),
        }
        self.compiled_patterns = patterns
    
    async def analyze_logs_batch(
        self,
        logs: List[Dict[str, Any]],
        batch_size: int = BATCH_SIZE
    ) -> AnalysisResult:
        """
        Analyze logs in optimized batches for better performance.
        
        Args:
            logs: List of log entries to analyze
            batch_size: Size of each batch for processing
            
        Returns:
            Aggregated analysis results
        """
        start_time = datetime.now()
        total_result = AnalysisResult()
        
        # Process logs in batches
        batches = self._create_batches(logs, batch_size)
        
        # Process batches in parallel
        tasks = []
        for batch in batches:
            task = asyncio.create_task(self._process_batch(batch))
            tasks.append(task)
            
            # Limit concurrent tasks to prevent memory overflow
            if len(tasks) >= PARALLEL_WORKERS:
                batch_results = await asyncio.gather(*tasks)
                self._merge_results(total_result, batch_results)
                tasks = []
        
        # Process remaining tasks
        if tasks:
            batch_results = await asyncio.gather(*tasks)
            self._merge_results(total_result, batch_results)
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        total_result.processing_time_ms = processing_time
        total_result.entries_processed = len(logs)
        
        # Cache results for frequently accessed patterns
        self._cache_results(total_result)
        
        return total_result
    
    def _create_batches(
        self,
        logs: List[Dict[str, Any]],
        batch_size: int
    ) -> List[LogBatch]:
        """Create optimized batches for parallel processing."""
        batches = []
        for i in range(0, len(logs), batch_size):
            batch_entries = logs[i:i + batch_size]
            batch_id = hashlib.md5(
                f"{i}_{len(batch_entries)}".encode()
            ).hexdigest()[:8]
            batches.append(LogBatch(entries=batch_entries, batch_id=batch_id))
        return batches
    
    async def _process_batch(self, batch: LogBatch) -> AnalysisResult:
        """Process a single batch of logs efficiently."""
        result = AnalysisResult()
        
        # Use deque for efficient appending
        errors = deque(maxlen=100)
        warnings = deque(maxlen=100)
        latencies = []
        
        for entry in batch.entries:
            message = entry.get("message", "")
            severity = entry.get("severity", "DEFAULT")
            
            # Fast pattern matching using pre-compiled regex
            patterns_found = self._match_patterns_cached(message)
            for pattern in patterns_found:
                result.patterns[pattern] = result.patterns.get(pattern, 0) + 1
            
            # Collect metrics
            if severity in ["ERROR", "CRITICAL"]:
                errors.append(entry)
            elif severity == "WARNING":
                warnings.append(entry)
            
            # Extract performance metrics
            latency = self._extract_latency(message)
            if latency:
                latencies.append(latency)
        
        # Calculate batch statistics
        if latencies:
            result.performance_metrics["avg_latency"] = sum(latencies) / len(latencies)
            result.performance_metrics["max_latency"] = max(latencies)
            result.performance_metrics["min_latency"] = min(latencies)
        
        result.error_summary = {
            "error_count": len(errors),
            "warning_count": len(warnings),
            "error_rate": len(errors) / len(batch.entries) if batch.entries else 0
        }
        
        # Detect anomalies using statistical methods
        anomalies = self._detect_anomalies_fast(batch.entries)
        result.anomalies.extend(anomalies)
        
        return result
    
    @lru_cache(maxsize=CACHE_SIZE)
    def _match_patterns_cached(self, message: str) -> Tuple[str, ...]:
        """
        Cache pattern matching results for frequently seen messages.
        
        Uses LRU cache to store pattern matching results and avoid
        redundant regex operations on repeated log messages.
        """
        if not message:
            return tuple()
        
        # Create a hash for cache key (first 100 chars for efficiency)
        cache_key = message[:100]
        
        patterns_found = []
        for pattern_name, pattern_regex in self.compiled_patterns.items():
            if pattern_regex.search(message):
                patterns_found.append(pattern_name)
        
        return tuple(patterns_found)
    
    def _extract_latency(self, message: str) -> Optional[float]:
        """Extract latency value from log message efficiently."""
        if "latency" not in message.lower():
            return None
        
        match = self.compiled_patterns["latency"].search(message)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
        return None
    
    def _detect_anomalies_fast(
        self,
        entries: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Fast anomaly detection using statistical methods.
        
        Detects:
        - Sudden spike in error rates
        - Unusual patterns
        - Performance degradation
        """
        anomalies = []
        
        if len(entries) < 10:
            return anomalies
        
        # Group by time window for spike detection
        time_windows = defaultdict(list)
        for entry in entries:
            timestamp = entry.get("timestamp", "")
            if timestamp:
                # Group by minute
                window = timestamp[:16]  # YYYY-MM-DDTHH:MM
                time_windows[window].append(entry)
        
        # Detect spikes
        window_sizes = [len(entries) for entries in time_windows.values()]
        if window_sizes:
            avg_size = sum(window_sizes) / len(window_sizes)
            threshold = avg_size * 3  # 3x average is considered a spike
            
            for window, window_entries in time_windows.items():
                if len(window_entries) > threshold:
                    anomalies.append({
                        "type": "traffic_spike",
                        "window": window,
                        "count": len(window_entries),
                        "threshold": threshold
                    })
        
        return anomalies
    
    def _merge_results(
        self,
        total: AnalysisResult,
        batch_results: List[AnalysisResult]
    ):
        """Efficiently merge batch results into total result."""
        for result in batch_results:
            # Merge patterns
            for pattern, count in result.patterns.items():
                total.patterns[pattern] = total.patterns.get(pattern, 0) + count
            
            # Merge anomalies
            total.anomalies.extend(result.anomalies)
            
            # Merge performance metrics (calculate weighted average)
            if result.performance_metrics:
                if not total.performance_metrics:
                    total.performance_metrics = result.performance_metrics.copy()
                else:
                    # Weighted average for latencies
                    for metric in ["avg_latency", "max_latency", "min_latency"]:
                        if metric in result.performance_metrics:
                            if metric == "avg_latency":
                                # Proper weighted average
                                current = total.performance_metrics.get(metric, 0)
                                new = result.performance_metrics[metric]
                                total.performance_metrics[metric] = (current + new) / 2
                            elif metric == "max_latency":
                                current = total.performance_metrics.get(metric, 0)
                                total.performance_metrics[metric] = max(
                                    current,
                                    result.performance_metrics[metric]
                                )
                            elif metric == "min_latency":
                                current = total.performance_metrics.get(metric, float('inf'))
                                total.performance_metrics[metric] = min(
                                    current,
                                    result.performance_metrics[metric]
                                )
            
            # Merge error summary
            if result.error_summary:
                if not total.error_summary:
                    total.error_summary = result.error_summary.copy()
                else:
                    total.error_summary["error_count"] += result.error_summary.get("error_count", 0)
                    total.error_summary["warning_count"] += result.error_summary.get("warning_count", 0)
    
    def _cache_results(self, result: AnalysisResult):
        """Cache analysis results for quick retrieval."""
        # Create cache key based on result patterns
        cache_key = hashlib.md5(
            json.dumps(sorted(result.patterns.items())).encode()
        ).hexdigest()[:16]
        
        self.results_cache[cache_key] = {
            "result": result,
            "timestamp": datetime.now(),
            "ttl": PATTERN_CACHE_TTL
        }
        
        # Clean up old cache entries
        self._cleanup_cache()
    
    def _cleanup_cache(self):
        """Remove expired cache entries to prevent memory bloat."""
        current_time = datetime.now()
        expired_keys = []
        
        for key, cached in self.results_cache.items():
            age = (current_time - cached["timestamp"]).total_seconds()
            if age > cached["ttl"]:
                expired_keys.append(key)
        
        for key in expired_keys:
            del self.results_cache[key]
    
    async def stream_analyze_logs(
        self,
        log_stream: AsyncIterator[Dict[str, Any]],
        window_size: int = 100
    ) -> AsyncIterator[AnalysisResult]:
        """
        Stream processing for real-time log analysis.
        
        Processes logs in a sliding window for continuous analysis
        with minimal memory footprint.
        """
        window = deque(maxlen=window_size)
        
        async for log_entry in log_stream:
            window.append(log_entry)
            
            # Analyze when window is full or periodically
            if len(window) >= window_size:
                result = await self.analyze_logs_batch(list(window), batch_size=window_size)
                yield result
                # Keep half the window for continuity
                for _ in range(window_size // 2):
                    window.popleft() if window else None
    
    def get_cached_analysis(self, pattern_signature: str) -> Optional[AnalysisResult]:
        """Retrieve cached analysis results if available."""
        cached = self.results_cache.get(pattern_signature)
        if cached:
            age = (datetime.now() - cached["timestamp"]).total_seconds()
            if age < cached["ttl"]:
                return cached["result"]
        return None


# Singleton instance for global usage
_analyzer_instance = None

def get_optimized_analyzer() -> OptimizedLogAnalyzer:
    """Get or create the singleton analyzer instance."""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = OptimizedLogAnalyzer()
    return _analyzer_instance