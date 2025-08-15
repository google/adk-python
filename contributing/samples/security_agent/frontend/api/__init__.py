"""
API Client Package

This package provides specialized API clients for different services in the security agent.
"""

from .asset_inventory_client import (
    AssetInventoryClient,
    get_asset_inventory_client,
    asset_client,
    get_asset_summary,
    discover_assets,
    search_assets
)

__all__ = [
    'AssetInventoryClient',
    'get_asset_inventory_client', 
    'asset_client',
    'get_asset_summary',
    'discover_assets',
    'search_assets'
]