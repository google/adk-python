"""Service registry for managing modular services."""

from typing import Dict, Any, Optional, List, Type
import importlib
import logging
from datetime import datetime
import asyncio

from .service_config import ServiceConfig, ServiceDefinition, ServiceStatus
from .base_service import BaseService

logger = logging.getLogger(__name__)


class ServiceRegistry:
    """Central registry for all services."""
    
    def __init__(self, config: ServiceConfig, credentials=None, project_id=None):
        """Initialize service registry."""
        self.config = config
        self.credentials = credentials
        self.project_id = project_id
        self.services: Dict[str, BaseService] = {}
        self.routers: Dict[str, Any] = {}
        self._health_check_tasks: Dict[str, asyncio.Task] = {}
        
    def _load_module(self, module_path: str) -> Any:
        """Dynamically load a module."""
        try:
            parts = module_path.split('.')
            module = importlib.import_module('.'.join(parts[:-1]))
            return getattr(module, parts[-1])
        except Exception as e:
            logger.error(f"Failed to load module {module_path}: {e}")
            raise
    
    def _instantiate_service(self, service_def: ServiceDefinition) -> Optional[BaseService]:
        """Instantiate a service from its definition."""
        if not service_def.service_module:
            logger.warning(f"No service module defined for {service_def.name}")
            return None
            
        try:
            # Load the service class
            service_class = self._load_module(service_def.service_module)
            
            # Create instance with appropriate parameters
            if service_def.requires_gcp_auth:
                service = service_class(
                    service_name=service_def.name,
                    credentials=self.credentials,
                    project_id=self.project_id
                )
            else:
                service = service_class(service_name=service_def.name)
                
            return service
            
        except Exception as e:
            logger.error(f"Failed to instantiate service {service_def.name}: {e}")
            self.config.set_service_status(service_def.name, ServiceStatus.ERROR)
            return None
    
    def _load_router(self, service_def: ServiceDefinition) -> Optional[Any]:
        """Load API router for a service."""
        if not service_def.router_module:
            logger.warning(f"No router module defined for {service_def.name}")
            return None
            
        try:
            # Load the router module
            router_module = importlib.import_module(service_def.router_module)
            return getattr(router_module, 'router')
            
        except Exception as e:
            logger.error(f"Failed to load router for {service_def.name}: {e}")
            return None
    
    async def initialize_service(self, service_name: str) -> bool:
        """Initialize a single service."""
        service_def = self.config.get_service(service_name)
        if not service_def:
            logger.error(f"Service {service_name} not found in configuration")
            return False
        
        # Check if service is disabled
        if self.config.get_service_status(service_name) == ServiceStatus.DISABLED:
            logger.info(f"Service {service_name} is disabled, skipping initialization")
            return True
        
        # Check dependencies
        if not self.config.check_dependencies(service_name):
            logger.error(f"Dependencies not satisfied for service {service_name}")
            self.config.set_service_status(service_name, ServiceStatus.ERROR)
            return False
        
        try:
            # Instantiate service if needed
            if service_name not in self.services:
                service = self._instantiate_service(service_def)
                if not service:
                    return False
                self.services[service_name] = service
            
            # Start the service
            service = self.services[service_name]
            if await service.start():
                self.config.set_service_status(service_name, ServiceStatus.RUNNING)
                
                # Load router if available
                if service_def.router_module:
                    router = self._load_router(service_def)
                    if router:
                        self.routers[service_name] = {
                            'router': router,
                            'prefix': service_def.api_prefix,
                            'tags': service_def.tags
                        }
                
                # Start health check if configured
                if service_def.health_check:
                    self._start_health_check(service_name, service_def.health_check.interval_seconds)
                
                logger.info(f"Service {service_name} initialized successfully")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to initialize service {service_name}: {e}")
            self.config.set_service_status(service_name, ServiceStatus.ERROR)
            return False
    
    async def initialize_all_services(self) -> Dict[str, bool]:
        """Initialize all enabled services."""
        results = {}
        
        # Get services sorted by dependencies
        sorted_services = self._sort_services_by_dependencies()
        
        for service_name in sorted_services:
            service_def = self.config.get_service(service_name)
            if service_def and self.config.get_service_status(service_name) != ServiceStatus.DISABLED:
                results[service_name] = await self.initialize_service(service_name)
        
        return results
    
    def _sort_services_by_dependencies(self) -> List[str]:
        """Sort services by their dependencies (topological sort)."""
        # Build dependency graph
        graph = {}
        in_degree = {}
        
        for service_name in self.config.services:
            graph[service_name] = self.config.get_service_dependencies(service_name)
            in_degree[service_name] = 0
        
        # Calculate in-degrees
        for service_name in graph:
            for dep in graph[service_name]:
                if dep in in_degree:
                    in_degree[dep] += 1
        
        # Perform topological sort
        queue = [s for s in in_degree if in_degree[s] == 0]
        sorted_services = []
        
        while queue:
            service = queue.pop(0)
            sorted_services.append(service)
            
            for dependent in graph:
                if service in graph[dependent]:
                    in_degree[dependent] -= 1
                    if in_degree[dependent] == 0:
                        queue.append(dependent)
        
        return sorted_services
    
    async def shutdown_service(self, service_name: str) -> bool:
        """Shutdown a single service."""
        if service_name not in self.services:
            logger.warning(f"Service {service_name} not found in registry")
            return True
        
        try:
            # Stop health check if running
            if service_name in self._health_check_tasks:
                self._health_check_tasks[service_name].cancel()
                del self._health_check_tasks[service_name]
            
            # Stop the service
            service = self.services[service_name]
            if await service.stop():
                self.config.set_service_status(service_name, ServiceStatus.DISABLED)
                
                # Remove router if exists
                if service_name in self.routers:
                    del self.routers[service_name]
                
                logger.info(f"Service {service_name} shutdown successfully")
                return True
            else:
                return False
                
        except Exception as e:
            logger.error(f"Failed to shutdown service {service_name}: {e}")
            return False
    
    async def shutdown_all_services(self):
        """Shutdown all services."""
        # Shutdown in reverse dependency order
        sorted_services = list(reversed(self._sort_services_by_dependencies()))
        
        for service_name in sorted_services:
            if service_name in self.services:
                await self.shutdown_service(service_name)
    
    def get_service(self, service_name: str) -> Optional[BaseService]:
        """Get a service instance."""
        return self.services.get(service_name)
    
    def get_all_services(self) -> Dict[str, BaseService]:
        """Get all service instances."""
        return self.services.copy()
    
    def get_service_status(self, service_name: str) -> Dict[str, Any]:
        """Get detailed status of a service."""
        service = self.services.get(service_name)
        if service:
            return service.get_status()
        else:
            return {
                'service_name': service_name,
                'status': self.config.get_service_status(service_name).value,
                'initialized': False,
                'error': 'Service not loaded'
            }
    
    def get_all_statuses(self) -> Dict[str, Dict[str, Any]]:
        """Get status of all services."""
        statuses = {}
        
        for service_name in self.config.services:
            statuses[service_name] = self.get_service_status(service_name)
        
        return statuses
    
    def _start_health_check(self, service_name: str, interval: int):
        """Start periodic health check for a service."""
        async def health_check_loop():
            service = self.services[service_name]
            while True:
                try:
                    await asyncio.sleep(interval)
                    await service.check_health()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"Health check error for {service_name}: {e}")
        
        task = asyncio.create_task(health_check_loop())
        self._health_check_tasks[service_name] = task
    
    def get_available_routers(self) -> List[Dict[str, Any]]:
        """Get list of available API routers."""
        return list(self.routers.values())
    
    async def enable_service(self, service_name: str) -> bool:
        """Enable and start a service."""
        self.config.enable_service(service_name)
        return await self.initialize_service(service_name)
    
    async def disable_service(self, service_name: str) -> bool:
        """Disable and stop a service."""
        # Check if service can be disabled
        service_def = self.config.get_service(service_name)
        if service_def and service_def.required:
            raise ValueError(f"Cannot disable required service: {service_name}")
        
        # Shutdown the service
        if service_name in self.services:
            await self.shutdown_service(service_name)
        
        # Update configuration
        self.config.disable_service(service_name)
        return True
    
    async def restart_service(self, service_name: str) -> bool:
        """Restart a service."""
        service = self.services.get(service_name)
        if service:
            return await service.restart()
        else:
            # Try to initialize if not loaded
            return await self.initialize_service(service_name)