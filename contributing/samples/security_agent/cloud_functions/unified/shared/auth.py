"""
Authentication utilities for GCP services
"""

from typing import Any, Optional
import google.auth
from google.auth import exceptions as auth_exceptions
from google.auth.credentials import Credentials
import google.auth.transport.requests

from .config import Config


def get_authenticated_client(
    service: str,
    version: Optional[str] = None,
    scopes: Optional[list] = None
) -> Any:
    """
    Get authenticated client for GCP services

    Args:
        service: Service name (e.g., 'bigquery', 'iam', 'compute')
        version: API version (optional)
        scopes: OAuth scopes (optional)

    Returns:
        Authenticated service client
    """
    try:
        credentials, project = google.auth.default(scopes=scopes)
    except auth_exceptions.DefaultCredentialsError:
        if Config.ENABLE_SAMPLE_DATA:
            # Use anonymous credentials in sample mode to unblock local testing
            from google.auth.credentials import AnonymousCredentials

            credentials = AnonymousCredentials()
            project = Config.PROJECT_ID
        else:
            raise

    # Map service names to clients
    if service == 'bigquery':
        from google.cloud import bigquery
        return bigquery.Client(credentials=credentials, project=project)

    elif service == 'iam':
        from google.cloud import iam_admin_v1
        return iam_admin_v1.IAMClient(credentials=credentials)

    elif service == 'compute':
        from google.cloud import compute_v1
        return compute_v1.InstancesClient(credentials=credentials)

    elif service == 'storage':
        from google.cloud import storage
        return storage.Client(credentials=credentials, project=project)

    elif service == 'securitycenter':
        from google.cloud import securitycenter_v2
        return securitycenter_v2.SecurityCenterClient(credentials=credentials)

    elif service == 'resourcemanager':
        from google.cloud import resourcemanager_v3
        return resourcemanager_v3.ProjectsClient(credentials=credentials)

    else:
        raise ValueError(f"Unsupported service: {service}")


def get_credentials() -> Credentials:
    """Get default credentials"""
    credentials, _ = google.auth.default()
    return credentials
