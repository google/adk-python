"""Base service class for all modular services."""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import logging
from datetime import datetime

from .service_config import ServiceStatus

logger = logging.getLogger(__name__)


class BaseService(ABC):
    """Base class for all services in the modular architecture."""
    
    def __init__(self, service_name: str, credentials=None, project_id=None):
        """Initialize base service."""
        self.service_name = service_name
        self.credentials = credentials
        self.project_id = project_id
        self.status = ServiceStatus.NOT_CONFIGURED
        self.last_health_check = None
        self.health_status = {}
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """Initialize the service. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def shutdown(self) -> bool:
        """Shutdown the service gracefully. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check. Must be implemented by subclasses."""
        pass
    
    async def start(self) -> bool:
        """Start the service."""
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
        """Check service health and update status."""
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
        """Get current service status."""
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