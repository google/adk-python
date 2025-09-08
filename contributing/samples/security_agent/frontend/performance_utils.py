"""
Frontend Performance Utilities for GCP Security Agent
Streamlit optimizations for lazy loading, virtual scrolling, and progressive data loading
"""

import streamlit as st
import pandas as pd
import numpy as np
import asyncio
import time
import json
import threading
from typing import Any, Dict, List, Optional, Callable, Generator, Tuple
from dataclasses import dataclass
from functools import wraps, lru_cache
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
import weakref
from collections import defaultdict, deque
import pickle
import hashlib

logger = logging.getLogger(__name__)

@dataclass
class FrontendConfig:
    """Configuration for frontend performance optimizations"""
    # Virtual scrolling
    virtual_scroll_enabled: bool = True
    virtual_scroll_height: int = 400
    virtual_scroll_item_height: int = 30
    virtual_scroll_buffer: int = 5
    
    # Lazy loading
    lazy_loading_enabled: bool = True
    lazy_load_threshold: int = 100  # Load more when within 100px of bottom
    initial_load_size: int = 50
    lazy_load_batch_size: int = 25
    
    # Memoization
    memoization_enabled: bool = True
    memo_max_size: int = 1000
    memo_ttl: int = 300  # 5 minutes
    
    # Progressive loading
    progressive_loading_enabled: bool = True
    progressive_chunk_size: int = 10
    progressive_delay: float = 0.1  # seconds between chunks
    
    # Rendering optimization
    render_batch_size: int = 100
    render_debounce_delay: float = 0.5
    enable_component_caching: bool = True
    
    # Data optimization
    max_table_rows: int = 1000
    enable_data_sampling: bool = True
    sample_threshold: int = 10000

class PerformanceMonitor:
    """Monitor frontend performance metrics"""
    
    def __init__(self):
        self._metrics = defaultdict(list)
        self._timers = {}
        self._lock = threading.RLock()
    
    def start_timer(self, operation: str) -> str:
        """Start timing an operation"""
        timer_id = f"{operation}_{time.time()}"
        with self._lock:
            self._timers[timer_id] = time.time()
        return timer_id
    
    def end_timer(self, timer_id: str) -> float:
        """End timing and record duration"""
        with self._lock:
            if timer_id not in self._timers:
                return 0.0
            
            duration = time.time() - self._timers[timer_id]
            operation = timer_id.split('_')[0]
            self._metrics[f"{operation}_duration"].append(duration)
            del self._timers[timer_id]
            return duration
    
    def record_metric(self, name: str, value: float):
        """Record a custom metric"""
        with self._lock:
            self._metrics[name].append(value)
    
    def get_stats(self, operation: str = None) -> Dict[str, Any]:
        """Get performance statistics"""
        with self._lock:
            if operation:
                durations = self._metrics.get(f"{operation}_duration", [])
                if not durations:
                    return {}
                
                return {
                    'count': len(durations),
                    'avg': sum(durations) / len(durations),
                    'min': min(durations),
                    'max': max(durations),
                    'p95': sorted(durations)[int(len(durations) * 0.95)] if durations else 0
                }
            
            # All metrics
            stats = {}
            for metric_name, values in self._metrics.items():
                if values and 'duration' in metric_name:
                    stats[metric_name] = {
                        'count': len(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values)
                    }
            
            return stats

class SmartMemoizer:
    """Advanced memoization with TTL and LRU eviction"""
    
    def __init__(self, max_size: int = 1000, ttl: int = 300):
        self.max_size = max_size
        self.ttl = ttl
        self._cache = {}
        self._access_times = {}
        self._lock = threading.RLock()
    
    def _generate_key(self, func: Callable, args: tuple, kwargs: dict) -> str:
        """Generate cache key for function call"""
        key_data = {
            'func': func.__name__,
            'module': func.__module__,
            'args': str(args),
            'kwargs': str(sorted(kwargs.items()))
        }
        key_str = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_str.encode()).hexdigest()
    
    def get(self, func: Callable, args: tuple, kwargs: dict) -> Tuple[Any, bool]:
        """Get cached result"""
        cache_key = self._generate_key(func, args, kwargs)
        
        with self._lock:
            if cache_key not in self._cache:
                return None, False
            
            entry = self._cache[cache_key]
            
            # Check TTL
            if time.time() - entry['timestamp'] > self.ttl:
                del self._cache[cache_key]
                self._access_times.pop(cache_key, None)
                return None, False
            
            # Update access time
            self._access_times[cache_key] = time.time()
            return entry['result'], True
    
    def set(self, func: Callable, args: tuple, kwargs: dict, result: Any):
        """Cache result"""
        cache_key = self._generate_key(func, args, kwargs)
        
        with self._lock:
            # Evict oldest entries if at capacity
            while len(self._cache) >= self.max_size:
                oldest_key = min(self._access_times.items(), key=lambda x: x[1])[0]
                del self._cache[oldest_key]
                del self._access_times[oldest_key]
            
            # Cache entry
            self._cache[cache_key] = {
                'result': result,
                'timestamp': time.time()
            }
            self._access_times[cache_key] = time.time()
    
    def clear(self):
        """Clear cache"""
        with self._lock:
            self._cache.clear()
            self._access_times.clear()

class VirtualScrollContainer:
    """Virtual scrolling implementation for large datasets"""
    
    def __init__(
        self,
        data: List[Any],
        item_height: int = 30,
        container_height: int = 400,
        buffer_size: int = 5
    ):
        self.data = data
        self.item_height = item_height
        self.container_height = container_height
        self.buffer_size = buffer_size
        
        self.visible_count = container_height // item_height
        self.total_items = len(data)
    
    def get_visible_range(self, scroll_top: int) -> Tuple[int, int, List[Any]]:
        """Get the range of items that should be visible"""
        start_index = max(0, (scroll_top // self.item_height) - self.buffer_size)
        end_index = min(
            self.total_items,
            start_index + self.visible_count + (2 * self.buffer_size)
        )
        
        visible_data = self.data[start_index:end_index]
        return start_index, end_index, visible_data
    
    def render_virtual_list(
        self,
        render_func: Callable[[Any, int], Any],
        key_prefix: str = "virtual_item"
    ):
        """Render virtual scrolling list in Streamlit"""
        # Use session state to track scroll position
        scroll_key = f"{key_prefix}_scroll"
        if scroll_key not in st.session_state:
            st.session_state[scroll_key] = 0
        
        # Calculate visible range
        start_idx, end_idx, visible_data = self.get_visible_range(
            st.session_state[scroll_key]
        )
        
        # Create container with fixed height
        container = st.container()
        
        with container:
            # Spacer for items above visible area
            if start_idx > 0:
                spacer_height = start_idx * self.item_height
                st.markdown(
                    f'<div style="height: {spacer_height}px;"></div>',
                    unsafe_allow_html=True
                )
            
            # Render visible items
            for i, item in enumerate(visible_data):
                actual_index = start_idx + i
                render_func(item, actual_index)
            
            # Spacer for items below visible area
            remaining_items = self.total_items - end_idx
            if remaining_items > 0:
                spacer_height = remaining_items * self.item_height
                st.markdown(
                    f'<div style="height: {spacer_height}px;"></div>',
                    unsafe_allow_html=True
                )
        
        # Scroll position controls (for demonstration)
        if st.button("Scroll Down", key=f"{key_prefix}_scroll_down"):
            st.session_state[scroll_key] = min(
                st.session_state[scroll_key] + (self.item_height * 10),
                (self.total_items - self.visible_count) * self.item_height
            )
            st.experimental_rerun()
        
        return len(visible_data)

class LazyLoader:
    """Lazy loading implementation for progressive data loading"""
    
    def __init__(
        self,
        data_source: Callable[[int, int], List[Any]],
        initial_size: int = 50,
        batch_size: int = 25
    ):
        self.data_source = data_source
        self.initial_size = initial_size
        self.batch_size = batch_size
        self.loaded_data = []
        self.total_available = None
        self.loading = False
    
    def initialize(self) -> List[Any]:
        """Load initial batch of data"""
        if not self.loaded_data:
            self.loaded_data = self.data_source(0, self.initial_size)
        return self.loaded_data
    
    def load_more(self) -> List[Any]:
        """Load next batch of data"""
        if self.loading:
            return self.loaded_data
        
        self.loading = True
        try:
            start_idx = len(self.loaded_data)
            new_data = self.data_source(start_idx, self.batch_size)
            self.loaded_data.extend(new_data)
            
            # Update total if we got less than requested (end of data)
            if len(new_data) < self.batch_size:
                self.total_available = len(self.loaded_data)
        
        finally:
            self.loading = False
        
        return self.loaded_data
    
    def has_more(self) -> bool:
        """Check if more data is available"""
        if self.total_available is not None:
            return len(self.loaded_data) < self.total_available
        return True  # Assume more data available if not determined
    
    def render_with_load_more(
        self,
        render_func: Callable[[List[Any]], Any],
        key_prefix: str = "lazy_load"
    ):
        """Render data with load more functionality"""
        # Initialize data
        current_data = self.initialize()
        
        # Render current data
        render_func(current_data)
        
        # Load more button
        if self.has_more() and not self.loading:
            if st.button("Load More", key=f"{key_prefix}_load_more"):
                with st.spinner("Loading more data..."):
                    self.load_more()
                st.experimental_rerun()
        
        if self.loading:
            st.info("Loading more data...")
        
        # Show stats
        st.caption(f"Showing {len(current_data)} items")
        
        return current_data

class ProgressiveRenderer:
    """Progressive rendering for large datasets"""
    
    def __init__(
        self,
        data: List[Any],
        chunk_size: int = 10,
        delay: float = 0.1
    ):
        self.data = data
        self.chunk_size = chunk_size
        self.delay = delay
    
    def render_progressive(
        self,
        render_func: Callable[[Any, int], Any],
        progress_container: Any = None,
        key_prefix: str = "progressive"
    ):
        """Render data progressively"""
        if progress_container is None:
            progress_container = st.empty()
        
        progress_bar = st.progress(0)
        total_items = len(self.data)
        rendered_items = 0
        
        # Render in chunks
        for i in range(0, total_items, self.chunk_size):
            chunk = self.data[i:i + self.chunk_size]
            
            # Render chunk
            for j, item in enumerate(chunk):
                actual_index = i + j
                render_func(item, actual_index)
                rendered_items += 1
            
            # Update progress
            progress = rendered_items / total_items
            progress_bar.progress(progress)
            
            # Small delay to prevent blocking
            if i + self.chunk_size < total_items:
                time.sleep(self.delay)
        
        progress_bar.empty()
        return rendered_items

class DataOptimizer:
    """Optimize data for frontend rendering"""
    
    @staticmethod
    def sample_large_dataset(
        data: pd.DataFrame,
        max_rows: int = 1000,
        strategy: str = "random"
    ) -> pd.DataFrame:
        """Sample large datasets for better performance"""
        if len(data) <= max_rows:
            return data
        
        if strategy == "random":
            return data.sample(n=max_rows)
        elif strategy == "head":
            return data.head(max_rows)
        elif strategy == "stratified" and 'category' in data.columns:
            # Stratified sampling by category
            return data.groupby('category').apply(
                lambda x: x.sample(min(len(x), max_rows // data['category'].nunique()))
            ).reset_index(drop=True)
        else:
            return data.head(max_rows)
    
    @staticmethod
    def optimize_datatypes(data: pd.DataFrame) -> pd.DataFrame:
        """Optimize pandas DataFrame data types for memory efficiency"""
        optimized = data.copy()
        
        for col in optimized.columns:
            col_type = optimized[col].dtype
            
            if col_type == 'object':
                # Try to convert to category if low cardinality
                unique_ratio = optimized[col].nunique() / len(optimized)
                if unique_ratio < 0.5:  # Less than 50% unique values
                    optimized[col] = optimized[col].astype('category')
            
            elif col_type in ['int64', 'int32']:
                # Downcast integers
                min_val = optimized[col].min()
                max_val = optimized[col].max()
                
                if min_val >= -128 and max_val <= 127:
                    optimized[col] = optimized[col].astype('int8')
                elif min_val >= -32768 and max_val <= 32767:
                    optimized[col] = optimized[col].astype('int16')
                elif min_val >= -2147483648 and max_val <= 2147483647:
                    optimized[col] = optimized[col].astype('int32')
            
            elif col_type in ['float64', 'float32']:
                # Downcast floats
                optimized[col] = pd.to_numeric(optimized[col], downcast='float')
        
        return optimized
    
    @staticmethod
    def create_summary_stats(data: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for large datasets"""
        summary = {
            'shape': data.shape,
            'memory_usage_mb': data.memory_usage(deep=True).sum() / 1024 / 1024,
            'dtypes': data.dtypes.to_dict(),
            'null_counts': data.isnull().sum().to_dict(),
            'numeric_summary': {},
            'categorical_summary': {}
        }
        
        # Numeric columns summary
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            summary['numeric_summary'][col] = {
                'mean': data[col].mean(),
                'std': data[col].std(),
                'min': data[col].min(),
                'max': data[col].max(),
                'median': data[col].median()
            }
        
        # Categorical columns summary
        categorical_cols = data.select_dtypes(include=['object', 'category']).columns
        for col in categorical_cols:
            value_counts = data[col].value_counts()
            summary['categorical_summary'][col] = {
                'unique_count': data[col].nunique(),
                'top_values': value_counts.head(5).to_dict(),
                'null_count': data[col].isnull().sum()
            }
        
        return summary

# Performance decorators
def timed_operation(operation_name: str = None):
    """Decorator to time operations"""
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            op_name = operation_name or func.__name__
            timer_id = monitor.start_timer(op_name)
            try:
                result = func(*args, **kwargs)
                duration = monitor.end_timer(timer_id)
                
                # Log slow operations
                if duration > 1.0:
                    logger.warning(f"Slow operation: {op_name} took {duration:.2f}s")
                
                return result
            except Exception as e:
                monitor.end_timer(timer_id)
                raise
        return wrapper
    return decorator

def memoized(max_size: int = 1000, ttl: int = 300):
    """Decorator for memoizing function results"""
    def decorator(func):
        memo = SmartMemoizer(max_size, ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            result, found = memo.get(func, args, kwargs)
            if found:
                return result
            
            result = func(*args, **kwargs)
            memo.set(func, args, kwargs, result)
            return result
        
        return wrapper
    return decorator

@st.cache_data(ttl=300, max_entries=100)
def cached_data_processing(data_hash: str, processing_func: str, *args, **kwargs):
    """Cache expensive data processing operations"""
    # This function signature allows Streamlit to cache based on data hash
    # The actual processing function name is passed as a string parameter
    pass

def render_optimized_table(
    data: pd.DataFrame,
    max_rows: int = 1000,
    enable_sampling: bool = True,
    virtual_scroll: bool = True
) -> None:
    """Render table with performance optimizations"""
    config = FrontendConfig()
    
    # Optimize data if too large
    if enable_sampling and len(data) > max_rows:
        st.warning(f"Dataset has {len(data)} rows. Showing sample of {max_rows} rows.")
        data = DataOptimizer.sample_large_dataset(data, max_rows)
    
    # Optimize data types
    data = DataOptimizer.optimize_datatypes(data)
    
    if virtual_scroll and len(data) > config.virtual_scroll_buffer:
        # Use virtual scrolling for large datasets
        virtual_container = VirtualScrollContainer(
            data.to_dict('records'),
            config.virtual_scroll_item_height,
            config.virtual_scroll_height
        )
        
        def render_row(row_data, index):
            cols = st.columns(len(data.columns))
            for i, (col, value) in enumerate(row_data.items()):
                cols[i].text(str(value))
        
        virtual_container.render_virtual_list(render_row, "table_virtual")
    else:
        # Standard table rendering
        st.dataframe(data, use_container_width=True)

def render_progressive_content(
    data_generator: Generator[Any, None, None],
    render_func: Callable[[Any], None],
    chunk_size: int = 10
) -> None:
    """Render content progressively to avoid blocking"""
    config = FrontendConfig()
    
    if not config.progressive_loading_enabled:
        # Render all at once
        for item in data_generator:
            render_func(item)
        return
    
    # Progressive rendering
    container = st.container()
    progress_bar = st.progress(0)
    
    rendered_count = 0
    chunk = []
    
    for item in data_generator:
        chunk.append(item)
        
        if len(chunk) >= chunk_size:
            # Render chunk
            with container:
                for chunk_item in chunk:
                    render_func(chunk_item)
            
            rendered_count += len(chunk)
            chunk = []
            
            # Update progress (estimate)
            progress_bar.progress(min(rendered_count / 1000, 1.0))
            
            # Small delay
            time.sleep(config.progressive_delay)
    
    # Render remaining items
    if chunk:
        with container:
            for chunk_item in chunk:
                render_func(chunk_item)
    
    progress_bar.empty()

# Global instances
config = FrontendConfig()
monitor = PerformanceMonitor()
global_memoizer = SmartMemoizer(config.memo_max_size, config.memo_ttl)

# Streamlit component optimizations
def optimized_selectbox(
    label: str,
    options: List[Any],
    max_display: int = 1000,
    **kwargs
) -> Any:
    """Optimized selectbox for large option lists"""
    if len(options) > max_display:
        st.warning(f"Too many options ({len(options)}). Showing first {max_display}.")
        display_options = options[:max_display]
    else:
        display_options = options
    
    return st.selectbox(label, display_options, **kwargs)

def optimized_multiselect(
    label: str,
    options: List[Any],
    max_display: int = 1000,
    **kwargs
) -> List[Any]:
    """Optimized multiselect for large option lists"""
    if len(options) > max_display:
        st.warning(f"Too many options ({len(options)}). Use search or filter.")
        # Add search functionality
        search_term = st.text_input(f"Search {label.lower()}")
        if search_term:
            filtered_options = [
                opt for opt in options
                if search_term.lower() in str(opt).lower()
            ][:max_display]
        else:
            filtered_options = options[:max_display]
    else:
        filtered_options = options
    
    return st.multiselect(label, filtered_options, **kwargs)

# Performance monitoring utilities
def display_performance_metrics():
    """Display performance metrics in sidebar"""
    with st.sidebar:
        st.subheader("Performance Metrics")
        
        stats = monitor.get_stats()
        if stats:
            for operation, metrics in stats.items():
                with st.expander(f"{operation.replace('_', ' ').title()}"):
                    st.metric("Count", metrics['count'])
                    st.metric("Avg Duration", f"{metrics['avg']:.3f}s")
                    st.metric("Max Duration", f"{metrics['max']:.3f}s")
        else:
            st.info("No performance data yet")

def reset_performance_cache():
    """Reset all performance caches"""
    global_memoizer.clear()
    st.cache_data.clear()
    st.cache_resource.clear()
    st.success("Performance cache cleared")