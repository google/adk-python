"""
Service layer for ADK backend applications.

This module provides robust service layer implementations for external API integrations
with enterprise-grade features including rate limiting, circuit breakers, and graceful degradation.
"""

from .confluence_service import ConfluenceService

__all__ = ['ConfluenceService']