"""
Centralized page registry for the Streamlit application.

This module imports all page modules and organizes them into a dictionary
for easy access and registration in the main app. This approach simplifies
the main application file and makes it easier to add or remove pages.
"""

from . import (
    iam_analysis,
    asset_inventory,
    security_findings,
    network_security,
    compliance,
    settings
)

# A dictionary mapping page names to their corresponding modules.
# The `show_page` function from each module is used as the entry point.
PAGES = {
    "IAM Analysis": iam_analysis.show_page,
    "Asset Inventory": asset_inventory.show_page,
    "Security Findings": security_findings.show_page,
    "Network Security": network_security.show_page,
    "Compliance": compliance.show_page,
    "Settings": settings.show_page,
}