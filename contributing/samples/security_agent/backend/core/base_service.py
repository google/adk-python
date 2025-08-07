"""Base service class for all modular services.

This module defines the abstract base class that all services in the modular
architecture must inherit from. It provides a consistent interface for service
lifecycle management, health monitoring, and status reporting.

The BaseService class implements the Template Method pattern, defining the
overall structure of service operations while allowing subclasses to provide
specific implementations for initialization, shutdown, and health checks.

Key Features:
    - Standardized service lifecycle (start, stop, restart)
    - Built-in health monitoring capabilities
    - Automatic status tracking and reporting
    - Error handling and recovery mechanisms
    - Async/await support throughout

Example:
    Creating a custom service::
    
        class MyService(BaseService):
            async def initialize(self) -> bool:
                # Service-specific initialization
                self.client = await create_client()
                return True
                
            async def shutdown(self) -> bool:
                # Clean up resources
                await self.client.close()
                return True
                
            async def health_check(self) -> Dict[str, Any]:
                # Check service health
                return {
                    'healthy': await self.client.ping(),
                    'latency': 42
                }
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from .service_config import ServiceStatus

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Base class for all services in the modular architecture.
    
    This abstract base class defines the contract that all services must follow.
    It provides common functionality for service lifecycle management while
    requiring subclasses to implement service-specific behavior.
    
    The class manages service state transitions and ensures consistent behavior
    across all services in the system. State transitions are carefully controlled
    to prevent invalid states and ensure proper cleanup.
    
    Attributes:
        service_name (str): Unique identifier for the service
        credentials: GCP credentials for authenticated services
        project_id (str): GCP project ID for service operations
        status (ServiceStatus): Current service state
        last_health_check (datetime): Timestamp of last health check
        health_status (dict): Latest health check results
        _initialized (bool): Internal flag tracking initialization state
        
    State Diagram::
    
        NOT_CONFIGURED -> STARTING -> RUNNING -> STOPPING -> DISABLED
                             |           |            |
                             v           v            v
                           ERROR <-------+------------+
    """
    
    def __init__(self, service_name: str, credentials=None, project_id=None):
        """Initialize base service with common attributes.
        
        Sets up the foundational attributes that all services share. Subclasses
        should call super().__init__() before their own initialization.
        
        Args:
            service_name: Unique identifier for the service, used in logging
                         and status reporting
            credentials: Optional GCP credentials object. Required for services
                        that interact with Google Cloud APIs
            project_id: Optional GCP project ID. Required for GCP-integrated
                       services
                       
        Note:
            The service starts in NOT_CONFIGURED state and must be explicitly
            started using the start() method.
        """
        self.service_name = service_name
        self.credentials = credentials
        self.project_id = project_id
        self.status = ServiceStatus.NOT_CONFIGURED
        self.last_health_check = None
        self.health_status = {}
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service. Must be implemented by subclasses.
        
        This method should perform all necessary setup for the service,
        including:
        - Creating client connections
        - Loading configuration
        - Initializing caches or data structures
        - Performing initial health checks
        
        The method should be idempotent - calling it multiple times should
        not cause errors or resource leaks.
        
        Returns:
            bool: True if initialization succeeded, False otherwise
            
        Raises:
            Exception: Subclasses may raise exceptions which will be caught
                      and logged by the start() method
                      
        Example::
        
            async def initialize(self) -> bool:
                try:
                    self.client = SecureClient(self.credentials)
                    await self.client.connect()
                    self.cache = Cache(ttl=300)
                    return True
                except ConnectionError:
                    logger.error("Failed to connect to service")
                    return False
        """
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the service gracefully. Must be implemented by subclasses.
        
        This method should cleanly shut down the service, including:
        - Closing client connections
        - Flushing caches or buffers  
        - Canceling background tasks
        - Releasing resources
        
        The method should handle partial initialization gracefully - it may
        be called even if initialize() failed or was not called.
        
        Returns:
            bool: True if shutdown succeeded, False otherwise
            
        Raises:
            Exception: Subclasses may raise exceptions which will be caught
                      and logged by the stop() method
                      
        Example::
        
            async def shutdown(self) -> bool:
                try:
                    if hasattr(self, 'client'):
                        await self.client.close()
                    if hasattr(self, 'cache'):
                        await self.cache.flush()
                    return True
                except Exception as e:
                    logger.error(f"Error during shutdown: {e}")
                    return False
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check. Must be implemented by subclasses.
        
        This method should verify that the service is functioning correctly.
        Health checks should be lightweight and complete quickly (< 5 seconds).
        
        Common health checks include:
        - Verifying client connections are alive
        - Checking authentication is valid
        - Testing basic operations
        - Monitoring resource usage
        
        Returns:
            dict: Health status with at minimum a 'healthy' boolean key.
                 Additional keys may include metrics, latency, errors, etc.
                 
        Example Return::
        
            {
                'healthy': True,
                'latency_ms': 45,
                'connections': 3,
                'cache_hit_rate': 0.92,
                'last_error': None,
                'checks': {
                    'database': 'pass',
                    'api': 'pass',
                    'auth': 'pass'
                }
            }
            
        Note:
            This method is called periodically by the health monitoring system.
            It should not modify service state or perform heavy operations.
        """
        pass
    
    async def start(self) -> bool:
        """Start the service by calling initialize and updating status.
        
        This method implements the standard service startup sequence:
        1. Set status to STARTING
        2. Call the subclass initialize() method
        3. Update status based on initialization result
        4. Log the outcome
        
        This method is idempotent - calling it on an already running service
        will return True without re-initializing.
        
        Returns:
            bool: True if service started successfully or was already running,
                  False if initialization failed
                  
        Note:
            Exceptions from initialize() are caught and logged. The service
            status will be set to ERROR if initialization fails.
        """
        try:
            logger.info(f"Starting service: {self.service_name}")
            self.status = ServiceStatus.STARTING
            
            # Initialize the service
            if await self.initialize():
                self.status = ServiceStatus.RUNNING
                self._initialized = True
                logger.info(f"Service started successfully: {self.service_name}")
                return True
            else:
                self.status = ServiceStatus.ERROR
                logger.error(f"Failed to initialize service: {self.service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting service {self.service_name}: {e}")
            self.status = ServiceStatus.ERROR
            return False
    
    async def stop(self) -> bool:
        """Stop the service."""
        try:
            logger.info(f"Stopping service: {self.service_name}")
            self.status = ServiceStatus.STOPPING
            
            # Shutdown the service
            if await self.shutdown():
                self.status = ServiceStatus.DISABLED
                self._initialized = False
                logger.info(f"Service stopped successfully: {self.service_name}")
                return True
            else:
                logger.error(f"Failed to shutdown service: {self.service_name}")
                return False
                
        except Exception as e:
            logger.error(f"Error stopping service {self.service_name}: {e}")
            return False
    
    async def restart(self) -> bool:
        """Restart the service."""
        logger.info(f"Restarting service: {self.service_name}")
        
        # Stop if running
        if self.status == ServiceStatus.RUNNING:
            if not await self.stop():
                return False
        
        # Start the service
        return await self.start()
    
    async def check_health(self) -> Dict[str, Any]:
        """Check service health and update status based on results.
        
        This method wraps the subclass health_check() implementation and:
        1. Calls the health check
        2. Records the timestamp
        3. Updates service status based on health
        4. Handles and logs any exceptions
        
        The service status is automatically updated:
        - ERROR services become RUNNING if health check passes
        - RUNNING services become ERROR if health check fails
        
        Returns:
            dict: Health check results from the subclass implementation,
                  or an error dict if the health check raised an exception
                  
        Note:
            This method is called periodically by the health monitoring system
            and can also be called on-demand via the API.
        """
        try:
            self.health_status = await self.health_check()
            self.last_health_check = datetime.utcnow()
            
            # Update service status based on health
            if self.health_status.get('healthy', False):
                if self.status == ServiceStatus.ERROR:
                    self.status = ServiceStatus.RUNNING
            else:
                if self.status == ServiceStatus.RUNNING:
                    self.status = ServiceStatus.ERROR
                    
            return self.health_status
            
        except Exception as e:
            logger.error(f"Error checking health for service {self.service_name}: {e}")
            self.health_status = {
                'healthy': False,
                'error': str(e),
                'timestamp': datetime.utcnow().isoformat()
            }
            self.status = ServiceStatus.ERROR
            return self.health_status
    
    def get_status(self) -> Dict[str, Any]:
        """Get current service status with detailed information.
        
        Provides a comprehensive view of the service state including:
        - Current status (running, error, disabled, etc.)
        - Initialization state
        - Last health check results and timestamp
        - Service metadata
        
        Returns:
            dict: Complete service status information:
                - service_name: Unique service identifier
                - status: Current ServiceStatus value
                - initialized: Whether initialize() has succeeded
                - last_health_check: ISO timestamp of last health check
                - health_status: Latest health check results
                
        Example Return::
        
            {
                'service_name': 'security',
                'status': 'running',
                'initialized': True,
                'last_health_check': '2025-01-08T10:30:00Z',
                'health_status': {
                    'healthy': True,
                    'latency_ms': 45
                }
            }
        """
        return {
            'service_name': self.service_name,
            'status': self.status.value,
            'initialized': self._initialized,
            'last_health_check': self.last_health_check.isoformat() if self.last_health_check else None,
            'health_status': self.health_status
        }
    
    def is_healthy(self) -> bool:
        """Check if service is healthy."""
        return (
            self.status == ServiceStatus.RUNNING and 
            self.health_status.get('healthy', False)
        )
    
    def is_available(self) -> bool:
        """Check if service is available for use."""
        return self.status == ServiceStatus.RUNNING and self._initialized