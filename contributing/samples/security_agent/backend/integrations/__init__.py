"""
Integration Components for Phase 2 Advanced Security Features
===========================================================

This module provides integration clients for GCP services and APIs
used by the Phase 2 advanced security features.
"""

__version__ = "1.0.0"
__author__ = "Security Agent Team"

# GCP Integration clients
from .google_support_client import GoogleSupportClient
from .vpc_sc_client import VPCServiceControlsClient
from .gcp_billing_client import GCPBillingClient
from .gcp_resource_client import GCPResourceClient

__all__ = [
    "GoogleSupportClient",
    "VPCServiceControlsClient",
    "GCPBillingClient",
    "GCPResourceClient"
]