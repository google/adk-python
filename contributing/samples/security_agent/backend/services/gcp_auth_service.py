"""
GCP Authentication Service for secure token management and API authentication.

This service handles Google Cloud authentication including:
- Service account authentication
- Access token acquisition and refresh
- Secure token storage and management
- Fallback authentication strategies
"""

import json
import os
import subprocess
import time
from typing import Optional, Dict, Any, Tuple
from datetime import datetime, timedelta
import logging
import tempfile
import threading
from pathlib import Path

try:
    from google.oauth2 import service_account
    from google.auth.transport.requests import Request
    from google.auth import default
    import requests
    GCP_AUTH_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Google Cloud authentication libraries not available: {e}")
    GCP_AUTH_AVAILABLE = False

logger = logging.getLogger(__name__)

class GCPAuthenticationService:
    """
    Secure GCP authentication service with token management and refresh capabilities.
    
    Features:
    - Service account and Application Default Credentials (ADC) support
    - Automatic token refresh with configurable intervals
    - Secure in-memory token storage
    - Command-line gcloud authentication integration
    - Comprehensive error handling and logging
    """
    
    def __init__(self, project_id: str, service_account_path: Optional[str] = None):
        """
        Initialize the GCP Authentication Service.
        
        Args:
            project_id: GCP project ID
            service_account_path: Optional path to service account JSON file
        """
        self.project_id = project_id
        self.service_account_path = service_account_path
        self._access_token: Optional[str] = None
        self._token_expiry: Optional[datetime] = None
        self._credentials = None
        self._lock = threading.Lock()
        
        # Token refresh configuration
        self.token_refresh_margin = timedelta(minutes=5)  # Refresh 5 minutes before expiry
        self.max_retry_attempts = 3
        self.retry_delay = 2  # seconds
        
        logger.info(f"🔐 GCP AUTH: Initializing authentication service for project: {project_id}")
        
        if not GCP_AUTH_AVAILABLE:
            logger.warning("🚫 GCP AUTH: Google Cloud libraries not available - limited functionality")
            return
        
        # Initialize authentication
        self._initialize_authentication()
    
    def _initialize_authentication(self) -> None:
        """Initialize authentication with the best available method."""
        try:
            # Try service account authentication first
            if self.service_account_path and os.path.exists(self.service_account_path):
                logger.info(f"🔑 GCP AUTH: Using service account: {self.service_account_path}")
                self._initialize_service_account_auth()
            else:
                logger.info("🔑 GCP AUTH: Attempting Application Default Credentials (ADC)")
                self._initialize_default_credentials()
                
            # Test authentication
            self._test_authentication()
            
        except Exception as e:
            logger.error(f"❌ GCP AUTH: Failed to initialize authentication: {e}")
            logger.info("🔄 GCP AUTH: Falling back to gcloud command-line authentication")
    
    def _initialize_service_account_auth(self) -> None:
        """Initialize service account authentication."""
        try:
            with open(self.service_account_path, 'r') as f:
                service_account_info = json.load(f)
            
            self._credentials = service_account.Credentials.from_service_account_info(
                service_account_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            logger.info(f"✅ GCP AUTH: Service account authentication initialized")
            logger.info(f"🔐 GCP AUTH: Service account email: {service_account_info.get('client_email', 'unknown')}")
            
        except Exception as e:
            logger.error(f"❌ GCP AUTH: Service account authentication failed: {e}")
            raise
    
    def _initialize_default_credentials(self) -> None:
        """Initialize default credentials (ADC)."""
        try:
            self._credentials, project = default(scopes=['https://www.googleapis.com/auth/cloud-platform'])
            
            if project:
                logger.info(f"🔐 GCP AUTH: Default credentials project: {project}")
            
            logger.info("✅ GCP AUTH: Application Default Credentials initialized")
            
        except Exception as e:
            logger.error(f"❌ GCP AUTH: Default credentials failed: {e}")
            raise
    
    def _test_authentication(self) -> None:
        """Test authentication by acquiring an access token."""
        try:
            token = self.get_access_token()
            if token:
                logger.info("✅ GCP AUTH: Authentication test successful")
                logger.info(f"🕒 GCP AUTH: Token expires at: {self._token_expiry}")
            else:
                raise Exception("Failed to acquire access token")
                
        except Exception as e:
            logger.error(f"❌ GCP AUTH: Authentication test failed: {e}")
            raise
    
    def get_access_token(self, force_refresh: bool = False) -> Optional[str]:
        """
        Get a valid access token with automatic refresh.
        
        Args:
            force_refresh: Force token refresh even if current token is valid
            
        Returns:
            Valid access token or None if authentication fails
        """
        with self._lock:
            try:
                # Check if token needs refresh
                if force_refresh or self._needs_token_refresh():
                    logger.info("🔄 GCP AUTH: Refreshing access token...")
                    self._refresh_access_token()
                
                return self._access_token
                
            except Exception as e:
                logger.error(f"❌ GCP AUTH: Failed to get access token: {e}")
                
                # Try fallback authentication
                return self._try_fallback_authentication()
    
    def _needs_token_refresh(self) -> bool:
        """Check if access token needs to be refreshed."""
        if not self._access_token or not self._token_expiry:
            return True
        
        # Refresh if token expires within the refresh margin
        return datetime.utcnow() >= (self._token_expiry - self.token_refresh_margin)
    
    def _refresh_access_token(self) -> None:
        """Refresh the access token using available credentials."""
        if not GCP_AUTH_AVAILABLE:
            raise Exception("Google Cloud authentication libraries not available")
        
        if self._credentials:
            # Use Google Cloud SDK credentials
            self._refresh_with_credentials()
        else:
            # Fall back to gcloud command
            self._refresh_with_gcloud()
    
    def _refresh_with_credentials(self) -> None:
        """Refresh token using Google Cloud SDK credentials."""
        try:
            # Refresh the credentials
            self._credentials.refresh(Request())
            
            self._access_token = self._credentials.token
            
            # Calculate expiry time
            if hasattr(self._credentials, 'expiry') and self._credentials.expiry:
                self._token_expiry = self._credentials.expiry.replace(tzinfo=None)
            else:
                # Default to 1 hour if expiry not available
                self._token_expiry = datetime.utcnow() + timedelta(hours=1)
            
            logger.info("✅ GCP AUTH: Token refreshed successfully with SDK credentials")
            logger.debug(f"🕒 GCP AUTH: New token expires at: {self._token_expiry}")
            
        except Exception as e:
            logger.error(f"❌ GCP AUTH: Token refresh with SDK credentials failed: {e}")
            raise
    
    def _refresh_with_gcloud(self) -> None:
        """Refresh token using gcloud command-line tool."""
        try:
            logger.info("🔄 GCP AUTH: Using gcloud for token refresh...")
            
            # Run gcloud auth print-access-token command
            result = subprocess.run([
                'gcloud', 'auth', 'print-access-token', 
                '--project', self.project_id
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode != 0:
                error_msg = result.stderr.strip() if result.stderr else "Unknown error"
                raise Exception(f"gcloud command failed: {error_msg}")
            
            self._access_token = result.stdout.strip()
            
            # Set expiry time (gcloud tokens typically expire in 1 hour)
            self._token_expiry = datetime.utcnow() + timedelta(hours=1)
            
            logger.info("✅ GCP AUTH: Token refreshed successfully with gcloud")
            logger.debug(f"🕒 GCP AUTH: New token expires at: {self._token_expiry}")
            
        except subprocess.TimeoutExpired:
            logger.error("❌ GCP AUTH: gcloud command timed out")
            raise Exception("gcloud authentication timed out")
        except FileNotFoundError:
            logger.error("❌ GCP AUTH: gcloud command not found")
            raise Exception("gcloud CLI not installed or not in PATH")
        except Exception as e:
            logger.error(f"❌ GCP AUTH: gcloud token refresh failed: {e}")
            raise
    
    def _try_fallback_authentication(self) -> Optional[str]:
        """Try fallback authentication methods."""
        logger.info("🔄 GCP AUTH: Trying fallback authentication methods...")
        
        fallback_methods = [
            self._try_gcloud_fallback,
            self._try_metadata_service,
        ]
        
        for method in fallback_methods:
            try:
                token = method()
                if token:
                    return token
            except Exception as e:
                logger.warning(f"🔄 GCP AUTH: Fallback method failed: {e}")
                continue
        
        logger.error("❌ GCP AUTH: All fallback authentication methods failed")
        return None
    
    def _try_gcloud_fallback(self) -> Optional[str]:
        """Try gcloud as a fallback authentication method."""
        try:
            result = subprocess.run([
                'gcloud', 'auth', 'print-access-token'
            ], capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                token = result.stdout.strip()
                self._access_token = token
                self._token_expiry = datetime.utcnow() + timedelta(hours=1)
                logger.info("✅ GCP AUTH: Fallback gcloud authentication successful")
                return token
            
        except Exception as e:
            logger.debug(f"🔄 GCP AUTH: gcloud fallback failed: {e}")
        
        return None
    
    def _try_metadata_service(self) -> Optional[str]:
        """Try Google Cloud metadata service for authentication."""
        try:
            # This works on Compute Engine, Cloud Run, etc.
            url = "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token"
            headers = {"Metadata-Flavor": "Google"}
            
            response = requests.get(url, headers=headers, timeout=10)
            
            if response.status_code == 200:
                token_data = response.json()
                token = token_data.get('access_token')
                expires_in = token_data.get('expires_in', 3600)
                
                if token:
                    self._access_token = token
                    self._token_expiry = datetime.utcnow() + timedelta(seconds=expires_in)
                    logger.info("✅ GCP AUTH: Metadata service authentication successful")
                    return token
            
        except Exception as e:
            logger.debug(f"🔄 GCP AUTH: Metadata service fallback failed: {e}")
        
        return None
    
    def get_auth_headers(self) -> Dict[str, str]:
        """
        Get authorization headers for GCP API calls.
        
        Returns:
            Dictionary with authorization header
        """
        token = self.get_access_token()
        if token:
            return {"Authorization": f"Bearer {token}"}
        else:
            logger.warning("🚫 GCP AUTH: No valid access token available")
            return {}
    
    def make_authenticated_request(self, 
                                 url: str, 
                                 method: str = 'GET',
                                 **kwargs) -> requests.Response:
        """
        Make an authenticated HTTP request to GCP APIs.
        
        Args:
            url: API endpoint URL
            method: HTTP method (GET, POST, etc.)
            **kwargs: Additional arguments for requests
            
        Returns:
            HTTP response object
        """
        headers = kwargs.pop('headers', {})
        headers.update(self.get_auth_headers())
        
        for attempt in range(self.max_retry_attempts):
            try:
                logger.debug(f"🌐 GCP API: Making {method} request to {url}")
                
                response = requests.request(method, url, headers=headers, **kwargs)
                
                # Handle token expiry
                if response.status_code == 401:
                    logger.warning("🔄 GCP AUTH: Token expired, refreshing...")
                    self.get_access_token(force_refresh=True)
                    headers.update(self.get_auth_headers())
                    continue
                
                return response
                
            except Exception as e:
                logger.error(f"❌ GCP API: Request failed (attempt {attempt + 1}): {e}")
                if attempt < self.max_retry_attempts - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
        
        raise Exception(f"Failed to make authenticated request after {self.max_retry_attempts} attempts")
    
    def search_all_resources(self, 
                           query: Optional[str] = None,
                           asset_types: Optional[list] = None,
                           page_size: int = 100) -> Dict[str, Any]:
        """
        Search all resources using the Cloud Asset API searchAllResources endpoint.
        
        Args:
            query: Search query string
            asset_types: List of asset types to filter
            page_size: Number of results per page
            
        Returns:
            API response with discovered resources
        """
        try:
            url = f"https://cloudasset.googleapis.com/v1/projects/{self.project_id}:searchAllResources"
            
            params = {
                'pageSize': page_size
            }
            
            if query:
                params['query'] = query
            
            if asset_types:
                params['assetTypes'] = ','.join(asset_types)
            
            logger.info(f"🔍 GCP ASSET API: Searching resources in project {self.project_id}")
            logger.debug(f"🔍 Query: {query}, Asset Types: {asset_types}")
            
            response = self.make_authenticated_request(url, params=params)
            
            if response.status_code == 200:
                data = response.json()
                logger.info(f"✅ GCP ASSET API: Found {len(data.get('results', []))} resources")
                return data
            else:
                error_msg = f"API call failed with status {response.status_code}: {response.text}"
                logger.error(f"❌ GCP ASSET API: {error_msg}")
                raise Exception(error_msg)
                
        except Exception as e:
            logger.error(f"❌ GCP ASSET API: Resource search failed: {e}")
            raise
    
    def is_authenticated(self) -> bool:
        """
        Check if the service is properly authenticated.
        
        Returns:
            True if authentication is available and working
        """
        try:
            token = self.get_access_token()
            return token is not None
        except Exception:
            return False
    
    def get_authentication_status(self) -> Dict[str, Any]:
        """
        Get detailed authentication status information.
        
        Returns:
            Dictionary with authentication status details
        """
        status = {
            "authenticated": False,
            "project_id": self.project_id,
            "auth_method": None,
            "token_valid": False,
            "token_expiry": None,
            "time_to_expiry": None,
            "error": None
        }
        
        try:
            # Check authentication
            if self._credentials:
                status["auth_method"] = "service_account" if self.service_account_path else "default_credentials"
            else:
                status["auth_method"] = "gcloud_cli"
            
            # Check token
            token = self.get_access_token()
            status["authenticated"] = token is not None
            status["token_valid"] = token is not None
            
            if self._token_expiry:
                status["token_expiry"] = self._token_expiry.isoformat()
                time_remaining = self._token_expiry - datetime.utcnow()
                status["time_to_expiry"] = str(time_remaining)
            
        except Exception as e:
            status["error"] = str(e)
        
        return status
    
    def refresh_token_if_needed(self) -> bool:
        """
        Refresh token if it's close to expiry.
        
        Returns:
            True if token was refreshed or is still valid, False otherwise
        """
        try:
            if self._needs_token_refresh():
                logger.info("🔄 GCP AUTH: Token needs refresh, refreshing...")
                self.get_access_token(force_refresh=True)
                return True
            else:
                logger.debug("✅ GCP AUTH: Token is still valid")
                return True
        except Exception as e:
            logger.error(f"❌ GCP AUTH: Token refresh failed: {e}")
            return False