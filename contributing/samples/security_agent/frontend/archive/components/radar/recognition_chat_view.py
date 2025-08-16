"""
Recognition Phase Chat View - Resource discovery and inventory interface.

This module implements the Recognition phase of RADAR, focusing on
discovering and cataloging all cloud resources.
"""

import streamlit as st
import logging
from typing import Dict, Any, List
import sys
import os

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from components.shared.chat_streaming_base import StreamingChatBase
from components.radar.radar_state_manager import radar_state_manager, RADARPhase
from unified_api_client import api_client

logger = logging.getLogger(__name__)


class RecognitionChatView(StreamingChatBase):
    """
    Recognition phase chat interface.
    
    This phase focuses on:
    - Complete resource discovery
    - Asset inventory
    - Resource relationship mapping
    - Anomaly detection
    """
    
    def __init__(self):
        """Initialize Recognition chat view."""
        super().__init__(
            phase_name="Recognition",
            phase_icon="🔍",
            phase_description="Discover and inventory all cloud resources in your environment"
        )
    
    def render_quick_actions(self):
        """Render Recognition-specific quick actions."""
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("📊 Full Inventory", use_container_width=True):
                self.run_full_inventory()
        
        with col2:
            if st.button("🔍 Discover Resources", use_container_width=True):
                self.discover_resources()
        
        with col3:
            if st.button("🗺️ Map Relationships", use_container_width=True):
                self.map_resource_relationships()
        
        with col4:
            if st.button("🚨 Find Anomalies", use_container_width=True):
                self.detect_anomalies()
        
        # Standard actions
        super().render_quick_actions()
    
    def render_context_panel(self):
        """Render Recognition-specific context panel."""
        context = radar_state_manager.get_context()
        
        if context:
            # Show discovery statistics
            st.markdown("### 📈 Discovery Statistics")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                total_resources = self.get_phase_context().get("total_resources", 0)
                st.metric("Total Resources", total_resources)
            
            with col2:
                resource_types = self.get_phase_context().get("resource_types", 0)
                st.metric("Resource Types", resource_types)
            
            with col3:
                anomalies = self.get_phase_context().get("anomalies_detected", 0)
                st.metric("Anomalies", anomalies)
            
            # Resource breakdown
            resource_breakdown = self.get_phase_context().get("resource_breakdown", {})
            if resource_breakdown:
                st.markdown("### 📦 Resource Breakdown")
                for resource_type, count in resource_breakdown.items():
                    st.write(f"- **{resource_type}:** {count}")
        else:
            st.info("No discovery results yet. Start by running a full inventory.")
    
    def generate_response(self, user_input: str) -> str:
        """
        Generate Recognition-specific response.
        
        Args:
            user_input: User's query
            
        Returns:
            Generated response
        """
        try:
            # Prepare context
            context = radar_state_manager.get_context()
            project_id = context.project_id if context else st.session_state.get('selected_project', 'default')
            
            # Call RADAR backend
            response = api_client.radar_chat({
                "query": user_input,
                "phase": "recognition",
                "project_id": project_id,
                "context": self.get_phase_context()
            })
            
            if response.get("success"):
                # Update phase context with results
                if "recognition_results" in response:
                    self.update_phase_context("latest_discovery", response["recognition_results"])
                
                # Update RADAR state
                if context and radar_state_manager.can_execute_phase(RADARPhase.RECOGNITION):
                    radar_state_manager.start_phase(RADARPhase.RECOGNITION)
                    radar_state_manager.complete_phase(
                        RADARPhase.RECOGNITION,
                        response.get("recognition_results", {})
                    )
                
                return response.get("response", "Discovery completed.")
            else:
                return f"Error: {response.get('error', 'Failed to process recognition query')}"
                
        except Exception as e:
            logger.error(f"Recognition response generation failed: {e}")
            return f"Error generating response: {str(e)}"
    
    def run_full_inventory(self):
        """Execute a full resource inventory."""
        with st.spinner("Running complete resource inventory..."):
            try:
                # Call asset inventory API
                response = api_client.get_assets()
                
                if response.get("success"):
                    assets = response.get("assets", [])
                    
                    # Update context
                    self.update_phase_context("total_resources", len(assets))
                    self.update_phase_context("resource_breakdown", self._categorize_assets(assets))
                    self.update_phase_context("latest_inventory", assets)
                    
                    # Add system message
                    self.add_message(
                        "system",
                        f"✅ Full inventory completed: {len(assets)} resources discovered"
                    )
                    
                    # Show summary
                    st.success(f"Discovered {len(assets)} resources")
                    
                    # Display breakdown
                    breakdown = self._categorize_assets(assets)
                    if breakdown:
                        st.markdown("### Resource Types Found:")
                        for resource_type, count in breakdown.items():
                            st.write(f"- {resource_type}: {count}")
                else:
                    st.error(f"Inventory failed: {response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Full inventory failed: {e}")
                st.error(f"Failed to run inventory: {str(e)}")
        
        st.rerun()
    
    def discover_resources(self):
        """Discover new resources."""
        with st.spinner("Discovering resources..."):
            try:
                # Call discovery API
                response = api_client.discover_resources({
                    "project_id": st.session_state.get('selected_project', 'default')
                })
                
                if response.get("success"):
                    discovered = response.get("resources", [])
                    
                    # Update context
                    self.update_phase_context("newly_discovered", discovered)
                    self.update_phase_context("discovery_timestamp", response.get("timestamp"))
                    
                    # Add message
                    self.add_message(
                        "system",
                        f"🔍 Discovery completed: {len(discovered)} new resources found"
                    )
                    
                    st.success(f"Discovered {len(discovered)} new resources")
                else:
                    st.error(f"Discovery failed: {response.get('error', 'Unknown error')}")
                    
            except Exception as e:
                logger.error(f"Resource discovery failed: {e}")
                st.error(f"Failed to discover resources: {str(e)}")
        
        st.rerun()
    
    def map_resource_relationships(self):
        """Map relationships between resources."""
        with st.spinner("Mapping resource relationships..."):
            try:
                # Get current inventory
                inventory = self.get_phase_context().get("latest_inventory", [])
                
                if not inventory:
                    st.warning("No inventory available. Run full inventory first.")
                    return
                
                # Analyze relationships (simplified)
                relationships = self._analyze_relationships(inventory)
                
                # Update context
                self.update_phase_context("resource_relationships", relationships)
                self.update_phase_context("relationship_count", len(relationships))
                
                # Add message
                self.add_message(
                    "system",
                    f"🗺️ Mapped {len(relationships)} resource relationships"
                )
                
                st.success(f"Mapped {len(relationships)} relationships")
                
                # Show sample relationships
                if relationships:
                    st.markdown("### Sample Relationships:")
                    for rel in relationships[:5]:
                        st.write(f"- {rel}")
                        
            except Exception as e:
                logger.error(f"Relationship mapping failed: {e}")
                st.error(f"Failed to map relationships: {str(e)}")
        
        st.rerun()
    
    def detect_anomalies(self):
        """Detect anomalies in resources."""
        with st.spinner("Detecting anomalies..."):
            try:
                # Get current inventory
                inventory = self.get_phase_context().get("latest_inventory", [])
                
                if not inventory:
                    st.warning("No inventory available. Run full inventory first.")
                    return
                
                # Detect anomalies (simplified)
                anomalies = self._detect_resource_anomalies(inventory)
                
                # Update context
                self.update_phase_context("anomalies_detected", len(anomalies))
                self.update_phase_context("anomaly_details", anomalies)
                
                # Add message
                if anomalies:
                    self.add_message(
                        "system",
                        f"🚨 Detected {len(anomalies)} anomalies requiring attention"
                    )
                    
                    st.warning(f"Found {len(anomalies)} anomalies")
                    
                    # Show anomalies
                    st.markdown("### Detected Anomalies:")
                    for anomaly in anomalies:
                        st.write(f"- {anomaly}")
                else:
                    self.add_message(
                        "system",
                        "✅ No anomalies detected"
                    )
                    st.success("No anomalies detected")
                    
            except Exception as e:
                logger.error(f"Anomaly detection failed: {e}")
                st.error(f"Failed to detect anomalies: {str(e)}")
        
        st.rerun()
    
    def _categorize_assets(self, assets: List[Dict]) -> Dict[str, int]:
        """
        Categorize assets by type.
        
        Args:
            assets: List of assets
            
        Returns:
            Dictionary of asset types and counts
        """
        categories = {}
        for asset in assets:
            asset_type = asset.get("assetType", "Unknown")
            categories[asset_type] = categories.get(asset_type, 0) + 1
        return categories
    
    def _analyze_relationships(self, inventory: List[Dict]) -> List[str]:
        """
        Analyze relationships between resources.
        
        Args:
            inventory: Resource inventory
            
        Returns:
            List of relationship descriptions
        """
        relationships = []
        
        # Simple relationship detection
        for resource in inventory:
            resource_name = resource.get("name", "Unknown")
            resource_type = resource.get("assetType", "Unknown")
            
            # Check for parent relationships
            if "parent" in resource:
                parent = resource["parent"]
                relationships.append(f"{resource_name} is child of {parent}")
            
            # Check for network relationships
            if "network" in resource:
                network = resource["network"]
                relationships.append(f"{resource_name} is in network {network}")
            
            # Check for IAM relationships
            if "serviceAccount" in resource:
                sa = resource["serviceAccount"]
                relationships.append(f"{resource_name} uses service account {sa}")
        
        return relationships
    
    def _detect_resource_anomalies(self, inventory: List[Dict]) -> List[str]:
        """
        Detect anomalies in resource inventory.
        
        Args:
            inventory: Resource inventory
            
        Returns:
            List of detected anomalies
        """
        anomalies = []
        
        for resource in inventory:
            resource_name = resource.get("name", "Unknown")
            
            # Check for test resources in production
            if any(keyword in resource_name.lower() for keyword in ["test", "temp", "demo"]):
                anomalies.append(f"Possible test resource in production: {resource_name}")
            
            # Check for resources without tags
            if not resource.get("labels"):
                anomalies.append(f"Resource without tags: {resource_name}")
            
            # Check for old resources
            create_time = resource.get("createTime")
            if create_time:
                # Simple age check (would need proper date parsing)
                anomalies.append(f"Old resource detected: {resource_name}")
        
        return anomalies[:10]  # Limit to 10 anomalies for display


def render_recognition_chat_view():
    """Render the Recognition phase chat view."""
    chat_view = RecognitionChatView()
    
    # Start phase if needed
    context = radar_state_manager.get_context()
    if context and radar_state_manager.can_execute_phase(RADARPhase.RECOGNITION):
        if context.phases[RADARPhase.RECOGNITION].status == "pending":
            radar_state_manager.start_phase(RADARPhase.RECOGNITION)
    
    # Render interface
    chat_view.render_chat_interface()