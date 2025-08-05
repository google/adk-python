"""
Startup Status Component for Enhanced Security Agent Frontend.

Provides a loading page with backend connection checking and progress indicators.
"""

import streamlit as st
import requests
import time
from typing import Dict, Any, Optional
import json


class StartupStatusChecker:
    """Handles backend connectivity checking and startup status display."""
    
    def __init__(self, backend_url: str = "http://localhost:8000"):
        self.backend_url = backend_url
        self.health_endpoint = f"{backend_url}/health"
        self.info_endpoint = f"{backend_url}"
        
    def check_backend_health(self, timeout: int = 3) -> Dict[str, Any]:
        """Check if backend is healthy and responsive."""
        try:
            response = requests.get(self.health_endpoint, timeout=timeout)
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "response_time": response.elapsed.total_seconds(),
                    "data": response.json() if response.content else {}
                }
            else:
                return {
                    "status": "unhealthy",
                    "status_code": response.status_code,
                    "response_time": response.elapsed.total_seconds()
                }
        except requests.exceptions.ConnectionError:
            return {"status": "connection_refused", "error": "Backend not running"}
        except requests.exceptions.Timeout:
            return {"status": "timeout", "error": "Backend response timeout"}
        except Exception as e:
            return {"status": "error", "error": str(e)}
    
    def get_backend_info(self, timeout: int = 3) -> Dict[str, Any]:
        """Get backend service information."""
        try:
            response = requests.get(self.info_endpoint, timeout=timeout)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"HTTP {response.status_code}"}
        except Exception as e:
            return {"error": str(e)}
    
    def render_startup_status(self, max_attempts: int = 30, check_interval: int = 2) -> bool:
        """
        Render startup status page with backend connection checking.
        
        Returns:
            bool: True if backend is ready, False if giving up
        """
        st.set_page_config(
            page_title="Enhanced Security Agent - Starting Up",
            page_icon="🔒",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
        
        # Header
        st.markdown("""
        <div style='text-align: center; padding: 2rem;'>
            <h1>🔒 Enhanced Security Agent</h1>
            <h3>Starting Up...</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Create containers for dynamic content
        status_container = st.container()
        progress_container = st.container()
        details_container = st.container()
        
        # Initialize session state for tracking
        if 'startup_attempt' not in st.session_state:
            st.session_state.startup_attempt = 0
        if 'backend_ready' not in st.session_state:
            st.session_state.backend_ready = False
        
        # Progress tracking
        attempt = st.session_state.startup_attempt
        progress = min(attempt / max_attempts, 1.0)
        
        with status_container:
            st.markdown("### 🔍 Backend Connection Status")
            
            # Check backend health
            health_status = self.check_backend_health()
            
            if health_status["status"] == "healthy":
                st.success("✅ Backend is healthy and ready!")
                st.session_state.backend_ready = True
                
                # Show backend info
                backend_info = self.get_backend_info()
                if "service_name" in backend_info:
                    st.info(f"🚀 Connected to: {backend_info.get('service_name', 'Security Agent Backend')}")
                
                with details_container:
                    with st.expander("🔧 Backend Details", expanded=False):
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.metric("Response Time", f"{health_status.get('response_time', 0):.3f}s")
                            st.metric("Status", "Healthy", delta="✅")
                        
                        with col2:
                            if backend_info and "error" not in backend_info:
                                st.json(backend_info)
                
                # Auto-redirect after showing success
                time.sleep(2)
                st.rerun()
                return True
                
            elif health_status["status"] == "connection_refused":
                st.error("❌ Backend is not running")
                st.info("💡 Make sure to start the backend with `./run.sh`")
                
            elif health_status["status"] == "timeout":
                st.warning("⏰ Backend is starting up...")
                st.info("🔄 Backend is responding slowly, please wait...")
                
            elif health_status["status"] == "unhealthy":
                st.warning(f"⚠️ Backend returned HTTP {health_status.get('status_code')}")
                st.info("🔄 Backend may still be initializing...")
                
            else:
                st.error(f"❌ Backend error: {health_status.get('error', 'Unknown error')}")
        
        with progress_container:
            st.markdown("### 📊 Startup Progress")
            
            # Progress bar
            progress_bar = st.progress(progress)
            st.text(f"Attempt {attempt + 1} of {max_attempts}")
            
            # Status messages based on progress
            if progress < 0.3:
                st.info("🔧 Backend services are starting up...")
            elif progress < 0.6:
                st.info("⚙️ Initializing service account authentication...")
            elif progress < 0.8:
                st.info("🔗 Setting up Cloud Trace integration...")
            else:
                st.warning("⏳ Taking longer than expected...")
        
        with details_container:
            if attempt > 5:  # Show details after a few attempts
                with st.expander("🛠️ Troubleshooting", expanded=False):
                    st.markdown("""
                    **If the backend is taking too long to start:**
                    
                    1. **Check if backend is running:**
                       ```bash
                       curl http://localhost:8000/health
                       ```
                    
                    2. **Check backend logs:**
                       ```bash
                       tail -f logs/backend.log
                       ```
                    
                    3. **Restart services:**
                       ```bash
                       ./run.sh --force
                       ```
                    
                    4. **Check port availability:**
                       ```bash
                       lsof -i :8000
                       ```
                    """)
        
        # Auto-refresh logic
        if attempt < max_attempts and not st.session_state.backend_ready:
            st.session_state.startup_attempt += 1
            time.sleep(check_interval)
            st.rerun()
        elif attempt >= max_attempts:
            st.error("❌ Backend failed to start after maximum attempts")
            st.info("🔧 Please check the backend manually and refresh this page")
            if st.button("🔄 Retry Connection", type="primary"):
                st.session_state.startup_attempt = 0
                st.rerun()
            return False
        
        return False
    
    def render_connection_status_sidebar(self):
        """Render connection status in sidebar for running application."""
        with st.sidebar:
            st.markdown("---")
            st.subheader("🔗 Backend Status")
            
            health_status = self.check_backend_health(timeout=1)
            
            if health_status["status"] == "healthy":
                st.success("✅ Connected")
                st.metric("Response Time", f"{health_status.get('response_time', 0):.3f}s")
            else:
                st.error("❌ Disconnected")
                if st.button("🔄 Reconnect"):
                    st.rerun()


def check_backend_ready(backend_url: str = "http://localhost:8000", timeout: int = 3) -> bool:
    """
    Quick check if backend is ready.
    
    Returns:
        bool: True if backend is ready, False otherwise
    """
    try:
        response = requests.get(f"{backend_url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False


def render_startup_screen_if_needed(backend_url: str = "http://localhost:8000") -> bool:
    """
    Render startup screen if backend is not ready.
    
    Returns:
        bool: True if should continue to main app, False if still loading
    """
    # Quick check first
    if check_backend_ready(backend_url):
        return True
    
    # If backend not ready, show startup screen
    checker = StartupStatusChecker(backend_url)
    return checker.render_startup_status()


if __name__ == "__main__":
    # Demo/test the startup status checker
    checker = StartupStatusChecker()
    checker.render_startup_status()