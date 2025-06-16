"""
Memory Information Tool

This module provides a tool for gathering memory information.
"""

import time
from typing import Any, Dict

import psutil


def get_memory_info() -> Dict[str, Any]:
    """
    Gather memory information including RAM and swap usage.

    Returns:
        Dict[str, Any]: Dictionary with memory information structured for ADK
    """
    try:
        # Get memory information
        memory = psutil.virtual_memory()
        swap = psutil.swap_memory()

        memory_info = {
            "total_memory": f"{memory.total / (1024 ** 3):.2f} GB",
            "available_memory": f"{memory.available / (1024 ** 3):.2f} GB",
            "used_memory": f"{memory.used / (1024 ** 3):.2f} GB",
            "memory_percentage": f"{memory.percent:.1f}%",
            "swap_total": f"{swap.total / (1024 ** 3):.2f} GB",
            "swap_used": f"{swap.used / (1024 ** 3):.2f} GB",
            "swap_percentage": f"{swap.percent:.1f}%",
        }

        # Calculate stats
        memory_usage = memory.percent
        swap_usage = swap.percent
        high_memory_usage = memory_usage > 80
        high_swap_usage = swap_usage > 80

        # Format for ADK tool return structure
        return {
            "result": memory_info,
            "stats": {
                "memory_usage_percentage": memory_usage,
                "swap_usage_percentage": swap_usage,
                "total_memory_gb": memory.total / (1024**3),
                "available_memory_gb": memory.available / (1024**3),
            },
            "additional_info": {
                "data_format": "dictionary",
                "collection_timestamp": time.time(),
                "performance_concern": (
                    "High memory usage detected" if high_memory_usage else None
                ),
                "swap_concern": "High swap usage detected" if high_swap_usage else None,
            },
        }
    except Exception as e:
        return {
            "result": {"error": f"Failed to gather memory information: {str(e)}"},
            "stats": {"success": False},
            "additional_info": {"error_type": str(type(e).__name__)},
        }
