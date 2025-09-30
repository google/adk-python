"""
Confluence Service Integration Layer

This module provides integration between the existing ConfluenceTool and the new
enterprise-grade ConfluenceService for enhanced reliability and performance.

Implements bridge pattern to maintain backwards compatibility while adding
robust service layer features from T024-T026.
"""

import asyncio
import logging
import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from .confluence_service import (
    ConfluenceService,
    RateLimitConfig,
    CircuitBreakerConfig,
    RetryConfig,
    create_confluence_service
)

logger = logging.getLogger(__name__)


@dataclass
class IntegrationConfig:
    """Configuration for Confluence service integration"""

    # Service layer settings
    enable_circuit_breaker: bool = True
    enable_rate_limiting: bool = True
    enable_retry_logic: bool = True
    enable_connection_pooling: bool = True

    # Fallback settings
    enable_cache_fallback: bool = True
    enable_degraded_mode: bool = True

    # Performance settings
    max_concurrent_requests: int = 5
    request_timeout: float = 30.0
    pool_size: int = 10


class ConfluenceServiceIntegration:
    """
    Integration layer that bridges the existing ConfluenceTool with the new
    enterprise-grade ConfluenceService.

    This class provides:
    - Seamless migration from simple tool to enterprise service
    - Backwards compatibility with existing ADK tool interface
    - Enhanced reliability features from T024-T026
    - Graceful degradation when service layer is unavailable
    """

    def __init__(
        self,
        confluence_url: Optional[str] = None,
        username: Optional[str] = None,
        api_token: Optional[str] = None,
        config: Optional[IntegrationConfig] = None,
        sqlite_tool=None
    ):
        """
        Initialize integration layer.

        Args:
            confluence_url: Confluence URL (from env if not provided)
            username: Username (from env if not provided)
            api_token: API token (from env if not provided)
            config: Integration configuration
            sqlite_tool: SQLite tool for caching compatibility
        """
        self.config = config or IntegrationConfig()
        self.sqlite_tool = sqlite_tool

        # Configuration from environment if not provided
        self.confluence_url = confluence_url or os.getenv("CONFLUENCE_URL", "")
        self.username = username or os.getenv("CONFLUENCE_USERNAME", "")
        self.api_token = api_token or os.getenv("CONFLUENCE_API_TOKEN", "")

        # Service layer components
        self.service: Optional[ConfluenceService] = None
        self._service_available = False

        # Fallback tool (original implementation)
        self._fallback_tool = None

        # Initialize service layer
        self._init_service_layer()

    def _init_service_layer(self) -> None:
        """Initialize the enterprise service layer."""
        if not all([self.confluence_url, self.username, self.api_token]):
            logger.warning("Confluence credentials not configured - service layer disabled")
            return

        try:
            # Configure service components based on integration settings
            rate_limit_config = None
            if self.config.enable_rate_limiting:
                rate_limit_config = RateLimitConfig(
                    max_requests=100,
                    time_window=60,
                    burst_limit=10
                )

            circuit_breaker_config = None
            if self.config.enable_circuit_breaker:
                circuit_breaker_config = CircuitBreakerConfig(
                    failure_threshold=5,
                    recovery_timeout=60,
                    success_threshold=3,
                    timeout=self.config.request_timeout
                )

            retry_config = None
            if self.config.enable_retry_logic:
                retry_config = RetryConfig(
                    max_retries=3,
                    base_delay=1.0,
                    max_delay=60.0,
                    exponential_base=2.0,
                    jitter=True
                )

            # Create service instance
            self.service = create_confluence_service(
                confluence_url=self.confluence_url,
                username=self.username,
                api_token=self.api_token,
                rate_limit_config=rate_limit_config,
                circuit_breaker_config=circuit_breaker_config,
                retry_config=retry_config,
                pool_size=self.config.pool_size
            )

            self._service_available = True
            logger.info("Enterprise Confluence service layer initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize service layer: {e}")
            self._service_available = False

    def _get_fallback_tool(self):
        """Get fallback tool instance if service layer is unavailable."""
        if self._fallback_tool is None:
            try:
                # Import here to avoid circular dependencies
                from ..contributing.samples.security_agent.agents._tools.confluence_tool import ConfluenceTool
                self._fallback_tool = ConfluenceTool(sqlite_tool=self.sqlite_tool)
                logger.info("Initialized fallback Confluence tool")
            except Exception as e:
                logger.error(f"Failed to initialize fallback tool: {e}")
                self._fallback_tool = None

        return self._fallback_tool

    async def search_documentation(
        self,
        query: str,
        spaces: Optional[List[str]] = None,
        limit: int = 50,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Search Confluence documentation with enterprise reliability.

        Args:
            query: Search query
            spaces: List of space keys to search
            limit: Maximum number of results
            **kwargs: Additional arguments for compatibility

        Returns:
            List of search results
        """
        # Try enterprise service layer first
        if self._service_available and self.service:
            try:
                logger.debug(f"Using enterprise service for search: {query}")

                results = await self.service.search_content(
                    query=query,
                    spaces=spaces,
                    limit=limit
                )

                # Convert service results to tool format for compatibility
                return self._convert_service_results(results, "search")

            except Exception as e:
                logger.warning(f"Enterprise service search failed, trying fallback: {e}")

                # Disable service temporarily if it's failing
                if "CircuitOpenError" in str(type(e)):
                    self._service_available = False

        # Fallback to original tool implementation
        if self.config.enable_degraded_mode:
            fallback_tool = self._get_fallback_tool()
            if fallback_tool:
                try:
                    logger.info("Using fallback tool for search")
                    return await fallback_tool.search_documentation(
                        query=query,
                        spaces=spaces,
                        limit=limit
                    )
                except Exception as e:
                    logger.error(f"Fallback search also failed: {e}")

        # Return empty results if all methods fail
        logger.error("All search methods failed")
        return []

    async def get_document(
        self,
        document_id: str,
        use_cache: bool = True,
        **kwargs
    ) -> Optional[Dict[str, Any]]:
        """
        Get document content with enterprise reliability.

        Args:
            document_id: Document ID
            use_cache: Whether to use cache
            **kwargs: Additional arguments for compatibility

        Returns:
            Document content
        """
        # Try enterprise service layer first
        if self._service_available and self.service:
            try:
                logger.debug(f"Using enterprise service for document: {document_id}")

                result = await self.service.get_page_content(
                    page_id=document_id,
                    expand='body.storage,body.view,metadata.labels,version'
                )

                # Convert service result to tool format for compatibility
                return self._convert_service_document(result)

            except Exception as e:
                logger.warning(f"Enterprise service document fetch failed, trying fallback: {e}")

                # Disable service temporarily if it's failing
                if "CircuitOpenError" in str(type(e)):
                    self._service_available = False

        # Fallback to original tool implementation
        if self.config.enable_degraded_mode:
            fallback_tool = self._get_fallback_tool()
            if fallback_tool:
                try:
                    logger.info("Using fallback tool for document")
                    return await fallback_tool.get_document(
                        document_id=document_id,
                        use_cache=use_cache
                    )
                except Exception as e:
                    logger.error(f"Fallback document fetch also failed: {e}")

        # Return None if all methods fail
        logger.error("All document fetch methods failed")
        return None

    async def get_security_context(
        self,
        gcp_service: str,
        query: str,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Get security context with enterprise reliability.

        Args:
            gcp_service: GCP service name
            query: Security query
            **kwargs: Additional arguments for compatibility

        Returns:
            Security context
        """
        # Try service layer first by searching for relevant documents
        if self._service_available and self.service:
            try:
                logger.debug(f"Using enterprise service for security context: {gcp_service}")

                # Search for relevant documentation
                search_query = f"{gcp_service} security {query}"
                documents = await self.service.search_content(
                    query=search_query,
                    limit=10
                )

                # Build security context
                context = {
                    'service': gcp_service,
                    'query': query,
                    'relevant_documents': self._convert_service_results(documents, "search"),
                    'recommendations': [],
                    'confidence_score': min(len(documents) / 10.0, 1.0),
                    'source': 'enterprise_service'
                }

                # Get detailed content for top documents
                for doc in documents[:3]:  # Top 3 most relevant
                    if doc.get('id'):
                        try:
                            full_doc = await self.service.get_page_content(doc['id'])
                            if full_doc:
                                recommendation = self._extract_recommendation(full_doc, gcp_service)
                                if recommendation:
                                    context['recommendations'].append(recommendation)
                        except Exception as e:
                            logger.warning(f"Failed to get detailed document {doc['id']}: {e}")

                return context

            except Exception as e:
                logger.warning(f"Enterprise service security context failed, trying fallback: {e}")

                if "CircuitOpenError" in str(type(e)):
                    self._service_available = False

        # Fallback to original tool implementation
        if self.config.enable_degraded_mode:
            fallback_tool = self._get_fallback_tool()
            if fallback_tool:
                try:
                    logger.info("Using fallback tool for security context")
                    return await fallback_tool.get_security_context(
                        gcp_service=gcp_service,
                        query=query
                    )
                except Exception as e:
                    logger.error(f"Fallback security context also failed: {e}")

        # Return minimal context if all methods fail
        logger.error("All security context methods failed")
        return {
            'service': gcp_service,
            'query': query,
            'relevant_documents': [],
            'recommendations': [],
            'confidence_score': 0.0,
            'source': 'fallback_minimal',
            'error': 'All retrieval methods failed'
        }

    def _convert_service_results(
        self,
        service_results: List[Dict[str, Any]],
        result_type: str
    ) -> List[Dict[str, Any]]:
        """Convert service layer results to tool format for compatibility."""
        converted = []

        for result in service_results:
            if result_type == "search":
                # Convert search results
                converted_result = {
                    'id': result.get('id'),
                    'title': result.get('title'),
                    'space_key': result.get('space_key'),
                    'url': result.get('url'),
                    'excerpt': result.get('excerpt', ''),
                    'last_modified': result.get('last_modified'),
                    'labels': result.get('labels', []),
                    'type': result.get('type', 'page'),
                    'status': result.get('status', 'current')
                }
            else:
                # Default conversion
                converted_result = result

            converted.append(converted_result)

        return converted

    def _convert_service_document(self, service_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert service layer document to tool format for compatibility."""
        if not service_doc:
            return None

        return {
            'id': service_doc.get('id'),
            'space_key': service_doc.get('space', {}).get('key'),
            'title': service_doc.get('title'),
            'content_storage': service_doc.get('body', {}).get('storage', {}).get('value', ''),
            'content_view': service_doc.get('body', {}).get('view', {}).get('value', ''),
            'labels': [label.get('name', '') for label in service_doc.get('metadata', {}).get('labels', {}).get('results', [])],
            'created_by': service_doc.get('created_by'),
            'created_at': service_doc.get('created_at'),
            'modified_by': service_doc.get('version', {}).get('by', {}).get('username'),
            'modified_at': service_doc.get('version', {}).get('when'),
            'version': service_doc.get('version', {}).get('number'),
            'url': service_doc.get('_links', {}).get('webui', '')
        }

    def _extract_recommendation(self, document: Dict, service: str) -> Optional[Dict]:
        """Extract security recommendation from document."""
        # Get content from either storage or view format
        content = ''
        if 'body' in document:
            body = document['body']
            if 'view' in body and 'value' in body['view']:
                content = body['view']['value']
            elif 'storage' in body and 'value' in body['storage']:
                content = body['storage']['value']

        if service.lower() in content.lower():
            return {
                'title': document.get('title'),
                'description': f"Security guidance from: {document.get('title')}",
                'document_ref': document.get('id'),
                'priority': 'medium',
                'url': document.get('_links', {}).get('webui', ''),
                'source': 'enterprise_service'
            }
        return None

    async def get_service_health(self) -> Dict[str, Any]:
        """Get comprehensive service health information."""
        health_data = {
            'integration_layer': 'active',
            'service_layer_available': self._service_available,
            'fallback_available': self._get_fallback_tool() is not None,
            'configuration': {
                'circuit_breaker_enabled': self.config.enable_circuit_breaker,
                'rate_limiting_enabled': self.config.enable_rate_limiting,
                'retry_logic_enabled': self.config.enable_retry_logic,
                'connection_pooling_enabled': self.config.enable_connection_pooling,
                'degraded_mode_enabled': self.config.enable_degraded_mode
            }
        }

        # Get service layer health if available
        if self._service_available and self.service:
            try:
                service_health = await self.service.health_check()
                health_data['service_layer_health'] = service_health
            except Exception as e:
                health_data['service_layer_error'] = str(e)
                self._service_available = False

        return health_data

    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics from both service and fallback."""
        stats = {
            'integration_layer': True,
            'service_cache': None,
            'fallback_cache': None
        }

        # Get service layer cache stats
        if self._service_available and self.service:
            try:
                stats['service_cache'] = self.service.get_service_metrics()
            except Exception as e:
                logger.warning(f"Failed to get service cache stats: {e}")

        # Get fallback cache stats
        fallback_tool = self._get_fallback_tool()
        if fallback_tool:
            try:
                stats['fallback_cache'] = fallback_tool.get_cache_stats()
            except Exception as e:
                logger.warning(f"Failed to get fallback cache stats: {e}")

        return stats

    async def clear_cache(self, document_id: Optional[str] = None) -> Dict[str, int]:
        """Clear cache from both service and fallback layers."""
        cleared = {
            'service_layer': 0,
            'fallback_layer': 0
        }

        # Clear service layer cache
        if self._service_available and self.service:
            try:
                # Service layer doesn't have clear_cache method, but we could add it
                logger.info("Service layer cache clearing not implemented")
            except Exception as e:
                logger.warning(f"Failed to clear service cache: {e}")

        # Clear fallback cache
        fallback_tool = self._get_fallback_tool()
        if fallback_tool:
            try:
                cleared['fallback_layer'] = fallback_tool.clear_cache(document_id)
            except Exception as e:
                logger.warning(f"Failed to clear fallback cache: {e}")

        return cleared

    async def close(self):
        """Clean up resources."""
        logger.info("Closing ConfluenceServiceIntegration")

        if self.service:
            try:
                await self.service.close()
            except Exception as e:
                logger.warning(f"Error closing service: {e}")

        # Fallback tool doesn't have async close method
        self._fallback_tool = None

        logger.info("ConfluenceServiceIntegration closed successfully")


# Factory function for easy integration creation
def create_confluence_integration(
    config: Optional[IntegrationConfig] = None,
    sqlite_tool=None,
    **kwargs
) -> ConfluenceServiceIntegration:
    """
    Factory function to create a configured ConfluenceServiceIntegration.

    Args:
        config: Integration configuration
        sqlite_tool: SQLite tool for caching compatibility
        **kwargs: Additional configuration options

    Returns:
        Configured ConfluenceServiceIntegration instance
    """
    return ConfluenceServiceIntegration(
        config=config,
        sqlite_tool=sqlite_tool,
        **kwargs
    )


# Compatibility function for existing ADK tool interface
async def query_confluence_documentation(
    query: str,
    spaces: Optional[List[str]] = None,
    gcp_service: Optional[str] = None,
    **kwargs
) -> Dict[str, Any]:
    """
    Query Confluence documentation with enterprise reliability.

    This function provides a drop-in replacement for the original
    query_confluence_documentation function with enhanced reliability.

    Args:
        query: Search query or evaluation request
        spaces: Optional list of Confluence spaces to search
        gcp_service: Optional GCP service for context
        **kwargs: Additional arguments

    Returns:
        Dictionary with search results or security context
    """
    integration = create_confluence_integration()

    try:
        if gcp_service:
            # Get security context for GCP service
            return await integration.get_security_context(gcp_service, query)
        else:
            # Perform general search
            results = await integration.search_documentation(query, spaces)
            return {
                'query': query,
                'results': results,
                'count': len(results),
                'source': 'enterprise_integration'
            }
    finally:
        await integration.close()


# Example usage and testing
if __name__ == "__main__":
    import os
    from dotenv import load_dotenv

    load_dotenv()

    async def test_integration():
        """Test the Confluence service integration."""
        integration = create_confluence_integration()

        try:
            # Health check
            health = await integration.get_service_health()
            print(f"Integration health: {health}")

            # Search test
            results = await integration.search_documentation("security", limit=5)
            print(f"Search results: {len(results)} found")

            # Security context test
            if results:
                context = await integration.get_security_context("GCP Storage", "encryption")
                print(f"Security context: {context.get('confidence_score', 0)} confidence")

            # Cache stats
            stats = integration.get_cache_stats()
            print(f"Cache stats: {stats}")

        except Exception as e:
            print(f"Test failed: {e}")
        finally:
            await integration.close()

    # Run test if executed directly
    asyncio.run(test_integration())