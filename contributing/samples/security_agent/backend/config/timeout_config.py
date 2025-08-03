"""Timeout configuration management with graceful degradation."""

import os
from typing import Dict, Any, Optional
from dataclasses import dataclass, asdict
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class OperationType(Enum):
    """Types of operations with different timeout requirements."""
    QUICK_CHAT = "quick_chat"
    STANDARD_ANALYSIS = "standard_analysis"
    COMPREHENSIVE_SCAN = "comprehensive_scan"
    DEEP_SCAN = "deep_scan"
    VULNERABILITY_SCAN = "vulnerability_scan"
    COMPLIANCE_CHECK = "compliance_check"
    CONFIGURATION_ANALYSIS = "configuration_analysis"
    DEPENDENCY_ANALYSIS = "dependency_analysis"

@dataclass
class TimeoutConfig:
    """Timeout configuration for different operation types."""
    # Frontend timeouts (seconds)
    frontend_quick_timeout: int = 30
    frontend_standard_timeout: int = 60
    frontend_long_timeout: int = 120
    
    # Backend operation timeouts (seconds)
    quick_chat_timeout: int = 30
    standard_analysis_timeout: int = 120
    comprehensive_scan_timeout: int = 600  # 10 minutes
    deep_scan_timeout: int = 1800  # 30 minutes
    
    # Individual scan component timeouts (seconds)
    vulnerability_scan_timeout: int = 300
    compliance_check_timeout: int = 180
    configuration_analysis_timeout: int = 120
    dependency_analysis_timeout: int = 90
    
    # Agent-specific timeouts (seconds)
    agent_chat_timeout: int = 45
    agent_tool_timeout: int = 60
    agent_long_operation_timeout: int = 300
    
    # Task management timeouts
    task_cleanup_interval: int = 3600  # 1 hour
    task_max_age_hours: int = 24
    task_progress_update_interval: int = 2
    
    # Graceful degradation settings
    enable_graceful_degradation: bool = True
    fallback_to_async_threshold: int = 30  # Fallback to async if operation exceeds this
    max_retry_attempts: int = 3
    retry_delay_seconds: int = 2
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_env(cls) -> 'TimeoutConfig':
        """Create timeout configuration from environment variables."""
        config = cls()
        
        # Override with environment variables if set
        env_mappings = {
            'FRONTEND_QUICK_TIMEOUT': 'frontend_quick_timeout',
            'FRONTEND_STANDARD_TIMEOUT': 'frontend_standard_timeout',
            'FRONTEND_LONG_TIMEOUT': 'frontend_long_timeout',
            'QUICK_CHAT_TIMEOUT': 'quick_chat_timeout',
            'STANDARD_ANALYSIS_TIMEOUT': 'standard_analysis_timeout',
            'COMPREHENSIVE_SCAN_TIMEOUT': 'comprehensive_scan_timeout',
            'DEEP_SCAN_TIMEOUT': 'deep_scan_timeout',
            'AGENT_CHAT_TIMEOUT': 'agent_chat_timeout',
            'AGENT_TOOL_TIMEOUT': 'agent_tool_timeout',
            'ENABLE_GRACEFUL_DEGRADATION': 'enable_graceful_degradation',
            'FALLBACK_TO_ASYNC_THRESHOLD': 'fallback_to_async_threshold',
            'MAX_RETRY_ATTEMPTS': 'max_retry_attempts'
        }
        
        for env_var, attr_name in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                try:
                    if attr_name == 'enable_graceful_degradation':
                        setattr(config, attr_name, env_value.lower() in ['true', '1', 'yes'])
                    else:
                        setattr(config, attr_name, int(env_value))
                    logger.info(f"Set {attr_name} = {getattr(config, attr_name)} from environment")
                except ValueError as e:
                    logger.warning(f"Invalid value for {env_var}: {env_value} - {e}")
        
        return config
    
    def get_timeout_for_operation(self, operation_type: OperationType) -> int:
        """Get appropriate timeout for operation type."""
        timeout_map = {
            OperationType.QUICK_CHAT: self.quick_chat_timeout,
            OperationType.STANDARD_ANALYSIS: self.standard_analysis_timeout,
            OperationType.COMPREHENSIVE_SCAN: self.comprehensive_scan_timeout,
            OperationType.DEEP_SCAN: self.deep_scan_timeout,
            OperationType.VULNERABILITY_SCAN: self.vulnerability_scan_timeout,
            OperationType.COMPLIANCE_CHECK: self.compliance_check_timeout,
            OperationType.CONFIGURATION_ANALYSIS: self.configuration_analysis_timeout,
            OperationType.DEPENDENCY_ANALYSIS: self.dependency_analysis_timeout
        }
        
        return timeout_map.get(operation_type, self.standard_analysis_timeout)
    
    def should_use_async(self, operation_type: OperationType, estimated_duration: Optional[int] = None) -> bool:
        """Determine if operation should use async processing."""
        if not self.enable_graceful_degradation:
            return False
            
        timeout = self.get_timeout_for_operation(operation_type)
        
        # Use async for operations that exceed fallback threshold
        if timeout > self.fallback_to_async_threshold:
            return True
            
        # Use async if estimated duration exceeds threshold
        if estimated_duration and estimated_duration > self.fallback_to_async_threshold:
            return True
            
        return False

class TimeoutManager:
    """Manager for timeout configuration and graceful degradation."""
    
    def __init__(self, config: Optional[TimeoutConfig] = None):
        """Initialize timeout manager.
        
        Args:
            config: Timeout configuration. If None, loads from environment.
        """
        self.config = config or TimeoutConfig.from_env()
        logger.info(f"Initialized TimeoutManager with config: {self.config.to_dict()}")
    
    def get_frontend_timeout(self, operation_complexity: str = "standard") -> int:
        """Get appropriate frontend timeout based on operation complexity.
        
        Args:
            operation_complexity: One of 'quick', 'standard', 'long'
            
        Returns:
            Timeout in seconds.
        """
        timeout_map = {
            'quick': self.config.frontend_quick_timeout,
            'standard': self.config.frontend_standard_timeout,
            'long': self.config.frontend_long_timeout
        }
        
        return timeout_map.get(operation_complexity, self.config.frontend_standard_timeout)
    
    def get_backend_timeout(self, operation_type: OperationType) -> int:
        """Get backend timeout for operation type."""
        return self.config.get_timeout_for_operation(operation_type)
    
    def should_fallback_to_async(self, operation_type: OperationType, estimated_duration: Optional[int] = None) -> bool:
        """Check if operation should fallback to async processing."""
        return self.config.should_use_async(operation_type, estimated_duration)
    
    def get_retry_config(self) -> Dict[str, int]:
        """Get retry configuration for failed operations."""
        return {
            'max_attempts': self.config.max_retry_attempts,
            'delay_seconds': self.config.retry_delay_seconds
        }
    
    def update_config(self, **kwargs) -> None:
        """Update timeout configuration dynamically.
        
        Args:
            **kwargs: Configuration parameters to update.
        """
        for key, value in kwargs.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Updated timeout config: {key} = {value}")
            else:
                logger.warning(f"Unknown timeout config parameter: {key}")
    
    def get_recommended_settings(self, deployment_type: str = "development") -> Dict[str, Any]:
        """Get recommended timeout settings for different deployment types.
        
        Args:
            deployment_type: One of 'development', 'staging', 'production'
            
        Returns:
            Dictionary of recommended settings.
        """
        settings = {
            'development': {
                'frontend_quick_timeout': 30,
                'frontend_standard_timeout': 60,
                'frontend_long_timeout': 120,
                'comprehensive_scan_timeout': 300,  # Shorter for dev
                'enable_graceful_degradation': True,
                'max_retry_attempts': 2
            },
            'staging': {
                'frontend_quick_timeout': 30,
                'frontend_standard_timeout': 90,
                'frontend_long_timeout': 180,
                'comprehensive_scan_timeout': 600,
                'enable_graceful_degradation': True,
                'max_retry_attempts': 3
            },
            'production': {
                'frontend_quick_timeout': 45,
                'frontend_standard_timeout': 120,
                'frontend_long_timeout': 300,
                'comprehensive_scan_timeout': 1200,  # Longer for production
                'deep_scan_timeout': 3600,  # 1 hour for production deep scans
                'enable_graceful_degradation': True,
                'max_retry_attempts': 3
            }
        }
        
        return settings.get(deployment_type, settings['development'])

# Global timeout manager instance
timeout_manager = TimeoutManager()