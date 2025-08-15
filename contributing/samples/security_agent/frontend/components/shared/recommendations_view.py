"""Recommendations view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api
from services.asset_data_service import AssetDataService


def render_recommendations_view():
    """Render the asset-driven security recommendations dashboard."""
    st.header("🎯 Asset-Driven Security Recommendations")
    st.write("Get security recommendations prioritized by asset impact and exposure.")
    
    # Initialize asset service for context
    asset_service = AssetDataService()
    project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
    
    # Get asset context for recommendation prioritization
    with st.spinner("Loading asset context for recommendation prioritization..."):
        asset_data = asset_service.get_asset_summary(project_id)
    
    # Asset-driven recommendation overview
    render_asset_recommendation_overview(asset_data)
    
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
    
    # Get asset-aware recommendations
    if st.button("🎯 Get Asset-Prioritized Recommendations") or 'recommendations_cache' in st.session_state:
        
        # Use cached data if available, enhanced with asset context
        if 'recommendations_cache' not in st.session_state:
            with st.spinner("Fetching asset-aware recommendations..."):
                response = simple_api.get_recommendations(priority)
                # Enhance recommendations with asset context
                if response.get('success') and asset_data.get('success'):
                    response = enhance_recommendations_with_assets(response, asset_data)
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
                
                # Asset-prioritized recommendations list
                st.subheader("📋 Asset-Prioritized Recommendations")
                
                # Sort recommendations by asset impact if available
                if 'asset_impact_score' in recommendations[0] if recommendations else {}:
                    recommendations = sorted(recommendations, key=lambda x: x.get('asset_impact_score', 0), reverse=True)
                    st.info("🎯 Recommendations sorted by asset impact and exposure")
                
                for i, rec in enumerate(recommendations):
                    with st.expander(f"🎯 {rec.get('title', 'Untitled Recommendation')}", 
                                   expanded=(i < 3)):  # Expand first 3
                        
                        # Priority badge
                        priority_color = {
                            "high": "🔴",
                            "medium": "🟡", 
                            "low": "🟢"
                        }.get(rec.get("priority", "unknown"), "⚪")
                        
                        # Enhanced priority display with asset context
                        st.markdown(f"**Priority:** {priority_color} {rec.get('priority', 'Unknown').title()}")
                        st.markdown(f"**Category:** {rec.get('category', 'General')}")
                        st.markdown(f"**Impact:** {rec.get('impact', 'Unknown')}")
                        st.markdown(f"**Effort:** {rec.get('effort', 'Unknown')}")
                        st.markdown(f"**Status:** {rec.get('status', 'Pending')}")
                        
                        # Asset-specific information
                        if 'asset_context' in rec:
                            asset_context = rec['asset_context']
                            st.markdown(f"**Asset Impact:** {asset_context.get('affected_assets', 0)} assets affected")
                            st.markdown(f"**Asset Types:** {', '.join(asset_context.get('asset_types', ['General']))}")
                            
                            impact_score = rec.get('asset_impact_score', 0)
                            if impact_score > 0:
                                impact_color = "inverse" if impact_score > 80 else "normal"
                                st.metric(
                                    "Asset Impact Score",
                                    f"{impact_score}/100",
                                    delta="Asset exposure risk",
                                    delta_color=impact_color
                                )
                        
                        # Description
                        st.markdown("**Description:**")
                        st.write(rec.get('description', 'No description available'))
                        
                        # Action items
                        actions = rec.get('actions', [])
                        if actions:
                            st.markdown("**Action Items:**")
                            for action in actions:
                                st.markdown(f"• {action}")
                        
                        # Compliance frameworks with asset correlation
                        frameworks = rec.get('compliance_frameworks', [])
                        if frameworks:
                            st.markdown("**Compliance Frameworks:**")
                            framework_badges = " ".join([f"`{fw}`" for fw in frameworks])
                            st.markdown(framework_badges)
                            
                            # Show which asset types are most affected by compliance
                            if 'asset_context' in rec and rec['asset_context'].get('compliance_sensitive_assets'):
                                sensitive_assets = rec['asset_context']['compliance_sensitive_assets']
                                st.markdown(f"**Compliance-Sensitive Assets:** {', '.join(sensitive_assets)}")
                        
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


def render_asset_recommendation_overview(asset_data: Dict[str, Any]):
    """Render asset-driven recommendation overview."""
    st.subheader("🎯 Asset-Driven Recommendation Overview")
    
    if asset_data.get('success') and asset_data.get('total_assets', 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_assets = asset_data.get('total_assets', 0)
            st.metric(
                "Assets for Recommendations",
                total_assets,
                help="Total assets considered for security recommendations"
            )
        
        with col2:
            high_risk_assets = asset_data.get('high_risk_count', 0)
            st.metric(
                "High-Priority Assets",
                high_risk_assets,
                delta=f"{(high_risk_assets/total_assets*100):.0f}% of total" if total_assets > 0 else "0%",
                delta_color="inverse" if high_risk_assets > 0 else "normal",
                help="Assets generating high-priority recommendations"
            )
        
        with col3:
            categories_count = len(asset_data.get('asset_categories', {}))
            complexity_score = min(100, categories_count * 10 + total_assets // 20)
            complexity_color = "inverse" if complexity_score > 70 else "normal"
            st.metric(
                "Recommendation Complexity",
                f"{complexity_score}/100",
                delta="Asset diversity impact",
                delta_color=complexity_color,
                help="Recommendation complexity based on asset diversity"
            )
        
        with col4:
            # Estimate recommendations based on asset types
            estimated_recs = len(asset_data.get('asset_categories', {})) * 2 + high_risk_assets
            st.metric(
                "Est. Recommendations",
                estimated_recs,
                delta="Asset-driven estimate",
                help="Estimated number of recommendations based on asset inventory"
            )
        
        # Asset category recommendation insights
        render_asset_category_recommendation_insights(asset_data)
    else:
        st.warning("🔍 No asset data for recommendation prioritization. Run asset discovery first.")
        if st.button("🚀 Discover Assets", use_container_width=True):
            st.session_state.page = "chat"
            st.session_state.suggested_query = "discover all assets in my GCP project"
            st.rerun()


def render_asset_category_recommendation_insights(asset_data: Dict[str, Any]):
    """Render recommendation insights by asset category."""
    st.subheader("📊 Asset Category Recommendation Insights")
    
    categories = asset_data.get('asset_categories', {})
    if categories:
        # Create recommendation priority matrix
        recommendation_data = []
        
        for category, count in categories.items():
            # Simulate recommendation priorities based on asset type
            if "compute" in category.lower():
                priority_score = 85  # High security needs
                rec_types = ["Access Control", "Patch Management", "Network Security"]
            elif "storage" in category.lower():
                priority_score = 90  # Critical data protection
                rec_types = ["Encryption", "Access Policies", "Backup Strategy"]
            elif "network" in category.lower():
                priority_score = 80  # Network security
                rec_types = ["Firewall Rules", "VPN Configuration", "Traffic Monitoring"]
            elif "database" in category.lower():
                priority_score = 95  # Critical data security
                rec_types = ["Data Encryption", "Access Control", "Audit Logging"]
            else:
                priority_score = 70  # Standard security
                rec_types = ["General Security", "Monitoring", "Compliance"]
            
            recommendation_data.append({
                "Asset Category": category,
                "Asset Count": count,
                "Priority Score": priority_score,
                "Est. Recommendations": len(rec_types),
                "Top Recommendation Types": ", ".join(rec_types[:2]),
                "Risk Level": "High" if priority_score > 85 else "Medium" if priority_score > 75 else "Low"
            })
        
        df = pd.DataFrame(recommendation_data)
        
        # Visualization
        col1, col2 = st.columns(2)
        
        with col1:
            # Priority score vs asset count
            fig = px.scatter(
                df,
                x="Asset Count",
                y="Priority Score",
                size="Est. Recommendations",
                color="Risk Level",
                hover_data=["Asset Category", "Top Recommendation Types"],
                title="Asset Recommendation Priority Matrix",
                color_discrete_map={
                    "High": "#ff4b4b",
                    "Medium": "#ffa500",
                    "Low": "#00cc88"
                }
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Priority breakdown by category
            fig_bar = px.bar(
                df.sort_values('Priority Score', ascending=True),
                x="Priority Score",
                y="Asset Category",
                orientation='h',
                color="Priority Score",
                title="Recommendation Priority by Asset Category",
                color_continuous_scale="Reds"
            )
            st.plotly_chart(fig_bar, use_container_width=True)
        
        # Detailed insights table
        st.dataframe(df, use_container_width=True)
        
        # Key insights
        st.subheader("💡 Key Insights")
        
        # Highest priority categories
        high_priority = df[df['Priority Score'] > 85].sort_values('Asset Count', ascending=False)
        if not high_priority.empty:
            st.error("🚨 High-priority asset categories requiring immediate attention:")
            for _, row in high_priority.head(3).iterrows():
                st.markdown(f"• **{row['Asset Category']}**: {row['Asset Count']} assets, {row['Est. Recommendations']} recommendations needed")
        
        # High asset count categories
        high_volume = df[df['Asset Count'] > 5].sort_values('Priority Score', ascending=False)
        if not high_volume.empty:
            st.warning("⚠️ High-volume asset categories:")
            for _, row in high_volume.head(3).iterrows():
                st.markdown(f"• **{row['Asset Category']}**: {row['Asset Count']} assets may require {row['Est. Recommendations']} types of recommendations")


def enhance_recommendations_with_assets(recommendations_response: Dict, asset_data: Dict) -> Dict:
    """Enhance recommendations with asset context and impact scoring."""
    if not recommendations_response.get('success') or not asset_data.get('success'):
        return recommendations_response
    
    recommendations = recommendations_response.get('recommendations', [])
    asset_categories = asset_data.get('asset_categories', {})
    total_assets = asset_data.get('total_assets', 0)
    
    enhanced_recommendations = []
    
    for rec in recommendations:
        enhanced_rec = rec.copy()
        
        # Add asset context based on recommendation category/type
        rec_category = rec.get('category', '').lower()
        rec_title = rec.get('title', '').lower()
        
        asset_context = {
            'affected_assets': 0,
            'asset_types': [],
            'compliance_sensitive_assets': [],
            'high_impact_categories': []
        }
        
        # Map recommendations to asset types
        for asset_type, count in asset_categories.items():
            asset_type_lower = asset_type.lower()
            
            # Check if recommendation applies to this asset type
            applies = False
            impact_multiplier = 1.0
            
            if 'iam' in rec_category or 'access' in rec_title:
                applies = True
                impact_multiplier = 1.5 if 'compute' in asset_type_lower else 1.0
            elif 'network' in rec_category or 'firewall' in rec_title:
                applies = 'network' in asset_type_lower or 'compute' in asset_type_lower
                impact_multiplier = 2.0
            elif 'storage' in rec_category or 'data' in rec_title:
                applies = 'storage' in asset_type_lower or 'database' in asset_type_lower
                impact_multiplier = 1.8
            elif 'encryption' in rec_title:
                applies = 'storage' in asset_type_lower or 'database' in asset_type_lower
                impact_multiplier = 2.0
            else:
                applies = True  # General recommendations apply to all
                impact_multiplier = 1.0
            
            if applies:
                asset_context['affected_assets'] += count
                asset_context['asset_types'].append(asset_type)
                
                if impact_multiplier > 1.5:
                    asset_context['high_impact_categories'].append(asset_type)
                
                # Mark compliance-sensitive assets
                if ('storage' in asset_type_lower or 'database' in asset_type_lower or 
                    'compute' in asset_type_lower):
                    asset_context['compliance_sensitive_assets'].append(asset_type)
        
        # Calculate asset impact score
        base_impact = asset_context['affected_assets'] / max(total_assets, 1) * 100
        high_impact_bonus = len(asset_context['high_impact_categories']) * 10
        compliance_bonus = len(asset_context['compliance_sensitive_assets']) * 5
        
        asset_impact_score = min(100, base_impact + high_impact_bonus + compliance_bonus)
        
        enhanced_rec['asset_context'] = asset_context
        enhanced_rec['asset_impact_score'] = asset_impact_score
        
        enhanced_recommendations.append(enhanced_rec)
    
    recommendations_response['recommendations'] = enhanced_recommendations
    recommendations_response['asset_enhancement'] = True
    
    return recommendations_response


def render_recommendations_summary_card():
    """Render enhanced recommendations summary card with asset integration."""
    with st.container():
        st.subheader("🎯 Asset-Driven Recommendations")
        
        # Get asset-aware recommendations
        project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
        asset_service = AssetDataService()
        
        try:
            asset_data = asset_service.get_asset_summary(project_id)
            response = simple_api.get_recommendations("high")
            
            # Enhance with asset context
            if response.get("success") and asset_data.get('success'):
                response = enhance_recommendations_with_assets(response, asset_data)
                recommendations = response.get("recommendations", [])[:3]  # Top 3
                
                if recommendations:
                    # Sort by asset impact score if available
                    if 'asset_impact_score' in recommendations[0]:
                        recommendations = sorted(recommendations, key=lambda x: x.get('asset_impact_score', 0), reverse=True)
                    
                    for i, rec in enumerate(recommendations):
                        impact_score = rec.get('asset_impact_score', 0)
                        affected_assets = rec.get('asset_context', {}).get('affected_assets', 0)
                        
                        priority_emoji = "🔴" if impact_score > 80 else "🟡" if impact_score > 60 else "🟢"
                        
                        with st.expander(f"{priority_emoji} {rec.get('title', 'Untitled')} (Impact: {impact_score:.0f}/100)", expanded=False):
                            st.write(rec.get('description', 'No description'))
                            st.markdown(f"**Impact:** {rec.get('impact', 'Unknown')}")
                            if affected_assets > 0:
                                st.markdown(f"**Affected Assets:** {affected_assets}")
                                asset_types = rec.get('asset_context', {}).get('asset_types', [])
                                if asset_types:
                                    st.markdown(f"**Asset Types:** {', '.join(asset_types[:3])}")
                else:
                    st.success("🎉 No high-priority asset recommendations!")
            else:
                st.error("Failed to load asset-aware recommendations")
        
        except Exception as e:
            st.error(f"Failed to load recommendations: {str(e)[:50]}...")
        
        if st.button("View All Asset Recommendations", key="view_all_recs"):
            st.session_state.page = "recommendations"
            st.rerun()