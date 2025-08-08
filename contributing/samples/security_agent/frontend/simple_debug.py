#!/usr/bin/env python3
"""Simple debug version to identify the exact frontend error."""

import streamlit as st
import logging
import traceback
import os
from datetime import datetime
from config import BACKEND_URL

# Setup logging
log_dir = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, 'simple_debug.log'), mode='a'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    logger.info("=== SIMPLE DEBUG APP STARTED ===")
    logger.info(f"Timestamp: {datetime.now()}")
    
    try:
        st.set_page_config(
            page_title="Debug Test",
            page_icon="🔍",
            layout="wide"
        )
        
        logger.info("✅ Page config set successfully")
        
        st.title("🔍 Frontend Debug Test")
        st.success("✅ Basic Streamlit rendering works!")
        
        logger.info("✅ Title and success message rendered")
        
        # Test imports one by one
        st.subheader("Import Tests")
        
        try:
            from startup_status import render_startup_screen_if_needed
            st.success("✅ startup_status imported")
            logger.info("✅ startup_status imported successfully")
        except Exception as e:
            st.error(f"❌ startup_status import failed: {e}")
            logger.error(f"❌ startup_status import failed: {e}")
            logger.error(traceback.format_exc())
            
        try:
            from api_client import api_client
            st.success("✅ api_client imported")
            logger.info("✅ api_client imported successfully")
        except Exception as e:
            st.error(f"❌ api_client import failed: {e}")
            logger.error(f"❌ api_client import failed: {e}")
            logger.error(traceback.format_exc())
            
        try:
            from components import render_dashboard_view
            st.success("✅ components imported")
            logger.info("✅ components imported successfully")
        except Exception as e:
            st.error(f"❌ components import failed: {e}")
            logger.error(f"❌ components import failed: {e}")
            logger.error(traceback.format_exc())
            
        # Test backend connection
        st.subheader("Backend Connection Test")
        try:
            import requests
            response = requests.get(f"{BACKEND_URL}/health", timeout=5)
            st.success(f"✅ Backend responds: {response.status_code}")
            logger.info(f"✅ Backend connection successful: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Backend connection failed: {e}")
            logger.error(f"❌ Backend connection failed: {e}")
            
        logger.info("=== SIMPLE DEBUG APP COMPLETED SUCCESSFULLY ===")
        
    except Exception as e:
        logger.error(f"CRITICAL ERROR in simple debug app: {e}")
        logger.error(traceback.format_exc())
        st.error(f"Critical error: {e}")
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()