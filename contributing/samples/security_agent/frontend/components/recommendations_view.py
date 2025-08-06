"""Recommendations view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any
from api_client import api_client


def render_recommendations_view():
    """Render the security recommendations dashboard."""
    st.header("🎯 Security Recommendations Dashboard")
    st.write("Get prioritized security recommendations for your GCP project.")
    
    # Priority selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        priority = st.selectbox(
            "Priority Level:",
            ["high", "medium", "low", "all"],
            index=0,
            help="Filter recommendations by priority level"
        )
    
    with col2:
        if st.button("🔄 Refresh Recommendations", type="primary"):
            st.session_state.pop('recommendations_cache', None)
    
    # Get recommendations
    if st.button("📊 Get Recommendations") or 'recommendations_cache' in st.session_state:
        
        # Use cached data if available
        if 'recommendations_cache' not in st.session_state:
            with st.spinner("Fetching recommendations..."):
                response = api_client.get_recommendations(priority)
                st.session_state.recommendations_cache = response
        else:
            response = st.session_state.recommendations_cache
        
        if response.get("success"):
            recommendations = response.get("recommendations", [])
            
            if recommendations:
                # Summary metrics
                st.subheader("📈 Summary")
                
                # Priority breakdown
                priority_counts = {}
                for rec in recommendations:
                    rec_priority = rec.get("priority", "unknown")
                    priority_counts[rec_priority] = priority_counts.get(rec_priority, 0) + 1
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Total Recommendations", len(recommendations))
                
                with col2:
                    st.metric("High Priority", priority_counts.get("high", 0), 
                             delta_color="inverse")
                
                with col3:
                    st.metric("Medium Priority", priority_counts.get("medium", 0))
                
                with col4:
                    st.metric("Low Priority", priority_counts.get("low", 0))
                
                # Priority distribution chart
                if len(priority_counts) > 1:
                    st.subheader("📊 Priority Distribution")
                    fig = px.pie(
                        values=list(priority_counts.values()),
                        names=list(priority_counts.keys()),
                        title="Recommendations by Priority",
                        color_discrete_map={
                            "high": "#ff4b4b",
                            "medium": "#ffa500", 
                            "low": "#00cc88"
                        }
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Recommendations list
                st.subheader("📋 Detailed Recommendations")
                
                for i, rec in enumerate(recommendations):
                    with st.expander(f"🎯 {rec.get('title', 'Untitled Recommendation')}", 
                                   expanded=(i < 3)):  # Expand first 3
                        
                        # Priority badge
                        priority_color = {
                            "high": "🔴",
                            "medium": "🟡", 
                            "low": "🟢"
                        }.get(rec.get("priority", "unknown"), "⚪")
                        
                        st.markdown(f"**Priority:** {priority_color} {rec.get('priority', 'Unknown').title()}")
                        st.markdown(f"**Category:** {rec.get('category', 'General')}")
                        st.markdown(f"**Impact:** {rec.get('impact', 'Unknown')}")
                        st.markdown(f"**Effort:** {rec.get('effort', 'Unknown')}")
                        st.markdown(f"**Status:** {rec.get('status', 'Pending')}")
                        
                        # Description
                        st.markdown("**Description:**")
                        st.write(rec.get('description', 'No description available'))
                        
                        # Action items
                        actions = rec.get('actions', [])
                        if actions:
                            st.markdown("**Action Items:**")
                            for action in actions:
                                st.markdown(f"• {action}")
                        
                        # Compliance frameworks
                        frameworks = rec.get('compliance_frameworks', [])
                        if frameworks:
                            st.markdown("**Compliance Frameworks:**")
                            framework_badges = " ".join([f"`{fw}`" for fw in frameworks])
                            st.markdown(framework_badges)
                        
                        # Mark as completed button
                        if st.button(f"✅ Mark as Completed", key=f"complete_{rec.get('id', i)}"):
                            st.success(f"Recommendation '{rec.get('title')}' marked as completed!")
                            # In a real implementation, this would update the backend
                            st.session_state.pop('recommendations_cache', None)  # Refresh cache
                            st.rerun()
            
            else:
                st.info("🎉 Great! No recommendations found for the selected priority level.")
        
        else:
            st.error(f"❌ Failed to fetch recommendations: {response.get('error', 'Unknown error')}")
    
    # Additional features
    st.markdown("---")
    st.subheader("🔧 Additional Features")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📊 Generate Report"):
            st.info("💡 Feature coming soon: Generate comprehensive security report")
    
    with col2:
        if st.button("📧 Email Report"):
            st.info("💡 Feature coming soon: Email report to stakeholders")


def render_recommendations_summary_card():
    """Render a compact recommendations summary card for the dashboard."""
    with st.container():
        st.subheader("🎯 Quick Recommendations")
        
        # Get top 3 high-priority recommendations
        response = api_client.get_recommendations("high")
        
        if response.get("success"):
            recommendations = response.get("recommendations", [])[:3]  # Top 3
            
            if recommendations:
                for rec in recommendations:
                    with st.expander(f"🔴 {rec.get('title', 'Untitled')}", expanded=False):
                        st.write(rec.get('description', 'No description'))
                        st.markdown(f"**Impact:** {rec.get('impact', 'Unknown')}")
            else:
                st.success("🎉 No high-priority recommendations!")
        else:
            st.error("Failed to load recommendations")
        
        if st.button("View All Recommendations", key="view_all_recs"):
            st.session_state.page = "recommendations"
            st.rerun()