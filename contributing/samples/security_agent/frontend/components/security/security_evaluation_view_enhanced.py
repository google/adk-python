"""Enhanced Security evaluation view with full asset integration."""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api
from services.asset_data_service import AssetDataService


def render_security_evaluation_view():
    """Render the asset-aware security evaluation dashboard."""
    st.header("🛡️ Asset-Aware Security Evaluation")
    st.write("Comprehensive security assessment correlated with your asset inventory.")
    
    # Initialize asset service
    asset_service = AssetDataService()
    project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
    
    # Get asset context first
    with st.spinner("Loading asset context..."):
        asset_data = asset_service.get_asset_summary(project_id)
    
    # Asset-aware header metrics
    render_asset_security_overview(asset_data)
    
    # Action buttons with asset context
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📊 Asset Security Analysis", type="primary"):
            run_asset_security_analysis(asset_service, project_id)
    
    with col2:
        if st.button("🔍 Asset Risk Assessment"):
            run_asset_risk_assessment(asset_service, project_id)
    
    with col3:
        if st.button("🔄 Full Asset Security Scan"):
            run_full_asset_security_scan(asset_service, project_id)
    
    # Display results
    if hasattr(st.session_state, 'asset_security_analysis'):
        render_asset_security_results()
    
    if hasattr(st.session_state, 'asset_risk_assessment'):
        render_asset_risk_results()
    
    if hasattr(st.session_state, 'full_asset_scan_results'):
        render_full_asset_scan_results()


def render_asset_security_overview(asset_data: Dict[str, Any]):
    """Render asset-aware security overview."""
    st.subheader("🎯 Asset Security Overview")
    
    if asset_data.get('success') and asset_data.get('total_assets', 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_assets = asset_data.get('total_assets', 0)
            st.metric(
                "Total Assets",
                total_assets,
                help="Total number of resources in your GCP project"
            )
        
        with col2:
            high_risk_count = asset_data.get('high_risk_count', 0)
            delta_color = "inverse" if high_risk_count > 0 else "normal"
            st.metric(
                "High-Risk Assets",
                high_risk_count,
                delta=f"{(high_risk_count/total_assets*100):.1f}% of total" if total_assets > 0 else "0%",
                delta_color=delta_color,
                help="Assets with critical security findings"
            )
        
        with col3:
            security_score = asset_data.get('security_score', 0)
            score_color = "normal" if security_score >= 80 else "inverse"
            st.metric(
                "Asset Security Score",
                f"{security_score}/100",
                delta="Asset-based calculation",
                delta_color=score_color,
                help="Security score calculated from real asset inventory"
            )
        
        with col4:
            categories_count = len(asset_data.get('asset_categories', {}))
            st.metric(
                "Asset Categories",
                categories_count,
                delta="Types monitored",
                help="Different types of assets being monitored"
            )
        
        # Asset category security breakdown
        render_asset_category_security_chart(asset_data)
    else:
        st.warning("🔍 No asset data available. Run asset discovery to enable security analysis.")
        if st.button("🚀 Discover Assets", use_container_width=True):
            st.session_state.page = "chat"
            st.session_state.suggested_query = "discover all assets in my GCP project"
            st.rerun()


def render_asset_category_security_chart(asset_data: Dict[str, Any]):
    """Render security analysis by asset category."""
    st.subheader("📊 Security Analysis by Asset Category")
    
    asset_categories = asset_data.get('asset_categories', {})
    if asset_categories:
        # Simulate security scores by category (in real implementation, get from backend)
        category_security = []
        for category, count in asset_categories.items():
            # Simulate risk levels based on asset type
            if "compute" in category.lower():
                risk_score = 65  # Higher risk for compute resources
            elif "storage" in category.lower():
                risk_score = 75  # Medium risk for storage
            elif "network" in category.lower():
                risk_score = 60  # Higher risk for network resources
            else:
                risk_score = 80  # Lower risk for other resources
            
            category_security.append({
                "Category": category,
                "Asset Count": count,
                "Security Score": risk_score,
                "Risk Level": "High" if risk_score < 70 else "Medium" if risk_score < 80 else "Low"
            })
        
        df = pd.DataFrame(category_security)
        
        # Create security heatmap
        fig = px.scatter(
            df,
            x="Asset Count",
            y="Security Score",
            size="Asset Count",
            color="Risk Level",
            hover_data=["Category"],
            title="Asset Security Risk Matrix",
            color_discrete_map={
                "High": "#ff4b4b",
                "Medium": "#ffa500",
                "Low": "#00cc88"
            }
        )
        
        fig.add_hline(y=80, line_dash="dash", line_color="green", annotation_text="Target Security Score")
        fig.add_hline(y=70, line_dash="dash", line_color="orange", annotation_text="Warning Threshold")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Category details table
        st.dataframe(df, use_container_width=True)


def run_asset_security_analysis(asset_service: AssetDataService, project_id: str):
    """Run security analysis with asset context."""
    with st.spinner("Analyzing security posture across all assets..."):
        # Get detailed asset data
        asset_data = asset_service.get_asset_summary(project_id)
        
        # Get security score from API
        security_response = simple_api.get_security_score()
        
        # Combine with asset context
        analysis_results = {
            "asset_data": asset_data,
            "security_response": security_response,
            "asset_security_correlation": correlate_security_with_assets(asset_data, security_response)
        }
        
        st.session_state.asset_security_analysis = analysis_results


def run_asset_risk_assessment(asset_service: AssetDataService, project_id: str):
    """Run risk assessment with asset correlation."""
    with st.spinner("Assessing risks across asset inventory..."):
        # Get asset data
        asset_data = asset_service.get_asset_summary(project_id)
        
        # Get enabled APIs with asset context
        apis_response = simple_api.get_enabled_apis()
        
        # Risk assessment combining assets and APIs
        risk_results = {
            "asset_risks": assess_asset_risks(asset_data),
            "api_risks": apis_response,
            "combined_risk_score": calculate_combined_risk_score(asset_data, apis_response)
        }
        
        st.session_state.asset_risk_assessment = risk_results


def run_full_asset_security_scan(asset_service: AssetDataService, project_id: str):
    """Run comprehensive security scan with full asset integration."""
    with st.spinner("Running comprehensive asset security scan..."):
        results = {}
        
        # Get comprehensive asset data
        results['asset_inventory'] = asset_service.get_asset_summary(project_id, force_refresh=True)
        
        # Get security metrics
        results['security_score'] = simple_api.get_security_score()
        results['enabled_apis'] = simple_api.get_enabled_apis()
        results['recommendations'] = simple_api.get_recommendations()
        results['compliance'] = simple_api.evaluate_compliance()
        
        # Asset-security correlations
        results['asset_security_mapping'] = create_asset_security_mapping(results)
        
        st.session_state.full_asset_scan_results = results


def correlate_security_with_assets(asset_data: Dict, security_data: Dict) -> Dict:
    """Correlate security findings with specific assets."""
    correlation = {
        "high_risk_asset_types": [],
        "secure_asset_types": [],
        "recommendations_by_asset_type": {}
    }
    
    if asset_data.get('success') and security_data.get('success'):
        asset_categories = asset_data.get('asset_categories', {})
        security_score = security_data.get('data', {}).get('overall_score', 0)
        
        # Simulate correlation (in real implementation, use actual security findings)
        for category, count in asset_categories.items():
            if "compute" in category.lower() or "network" in category.lower():
                if security_score < 70:
                    correlation["high_risk_asset_types"].append(category)
                    correlation["recommendations_by_asset_type"][category] = [
                        "Enable security monitoring",
                        "Review access controls",
                        "Update security policies"
                    ]
            else:
                correlation["secure_asset_types"].append(category)
    
    return correlation


def assess_asset_risks(asset_data: Dict) -> Dict:
    """Assess risks based on asset inventory."""
    risks = {
        "asset_exposure": {},
        "category_risks": {},
        "location_risks": {}
    }
    
    if asset_data.get('success'):
        # Analyze by category
        categories = asset_data.get('asset_categories', {})
        for category, count in categories.items():
            if count > 10:  # High asset count = higher exposure
                risks["category_risks"][category] = "High exposure due to asset count"
        
        # Analyze by location
        locations = asset_data.get('locations', {})
        for location, count in locations.items():
            if count > 5:
                risks["location_risks"][location] = f"Multiple assets ({count}) in single location"
    
    return risks


def calculate_combined_risk_score(asset_data: Dict, apis_data: Dict) -> int:
    """Calculate combined risk score from assets and APIs."""
    base_score = 100
    
    # Deduct points for high-risk factors
    if asset_data.get('success'):
        high_risk_assets = asset_data.get('high_risk_count', 0)
        total_assets = asset_data.get('total_assets', 1)
        risk_ratio = high_risk_assets / total_assets
        base_score -= int(risk_ratio * 30)
    
    if apis_data.get('success'):
        enabled_apis = apis_data.get('data', {}).get('enabled_apis', [])
        high_risk_apis = sum(1 for api in enabled_apis if api.get('risk_level') == 'high')
        base_score -= high_risk_apis * 5
    
    return max(0, base_score)


def create_asset_security_mapping(scan_results: Dict) -> Dict:
    """Create comprehensive asset-to-security mapping."""
    mapping = {
        "asset_security_matrix": {},
        "priority_assets": [],
        "security_coverage": {}
    }
    
    asset_data = scan_results.get('asset_inventory', {})
    if asset_data.get('success'):
        categories = asset_data.get('asset_categories', {})
        
        for category, count in categories.items():
            mapping["asset_security_matrix"][category] = {
                "count": count,
                "security_priority": "High" if count > 10 else "Medium",
                "monitoring_status": "Active",
                "last_scan": asset_data.get('timestamp', 'Unknown')
            }
    
    return mapping


def render_asset_security_results():
    """Render asset security analysis results."""
    st.subheader("📊 Asset Security Analysis Results")
    
    results = st.session_state.asset_security_analysis
    asset_data = results.get('asset_data', {})
    security_data = results.get('security_response', {})
    correlation = results.get('asset_security_correlation', {})
    
    if asset_data.get('success') and security_data.get('success'):
        # Security score with asset context
        col1, col2 = st.columns(2)
        
        with col1:
            overall_score = security_data.get('data', {}).get('overall_score', 0)
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=overall_score,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': f"Security Score ({asset_data.get('total_assets', 0)} Assets)"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 80
                    }
                }
            ))
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Asset risk breakdown
            st.subheader("🎯 Asset Risk Analysis")
            high_risk_types = correlation.get('high_risk_asset_types', [])
            secure_types = correlation.get('secure_asset_types', [])
            
            if high_risk_types:
                st.error(f"⚠️ High-risk asset types: {', '.join(high_risk_types)}")
            
            if secure_types:
                st.success(f"✅ Secure asset types: {', '.join(secure_types)}")
            
            # Asset-specific recommendations
            recommendations = correlation.get('recommendations_by_asset_type', {})
            if recommendations:
                st.subheader("💡 Asset-Specific Recommendations")
                for asset_type, recs in recommendations.items():
                    with st.expander(f"📋 {asset_type}"):
                        for rec in recs:
                            st.markdown(f"• {rec}")
    else:
        st.error("Failed to correlate security data with asset inventory")


def render_asset_risk_results():
    """Render asset risk assessment results."""
    st.subheader("🚨 Asset Risk Assessment")
    
    results = st.session_state.asset_risk_assessment
    asset_risks = results.get('asset_risks', {})
    combined_score = results.get('combined_risk_score', 0)
    
    # Combined risk score
    col1, col2, col3 = st.columns(3)
    
    with col1:
        score_color = "normal" if combined_score >= 80 else "inverse"
        st.metric(
            "Combined Risk Score",
            f"{combined_score}/100",
            delta="Asset + API correlation",
            delta_color=score_color
        )
    
    with col2:
        category_risks = len(asset_risks.get('category_risks', {}))
        st.metric(
            "Risk Categories",
            category_risks,
            delta="Asset types at risk"
        )
    
    with col3:
        location_risks = len(asset_risks.get('location_risks', {}))
        st.metric(
            "Location Risks",
            location_risks,
            delta="Geographic exposure"
        )
    
    # Detailed risk analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Category Risks")
        category_risks = asset_risks.get('category_risks', {})
        if category_risks:
            for category, risk in category_risks.items():
                st.warning(f"⚠️ **{category}**: {risk}")
        else:
            st.success("✅ No category-level risks detected")
    
    with col2:
        st.subheader("🌍 Location Risks")
        location_risks = asset_risks.get('location_risks', {})
        if location_risks:
            for location, risk in location_risks.items():
                st.warning(f"🌍 **{location}**: {risk}")
        else:
            st.success("✅ No location-based risks detected")


def render_full_asset_scan_results():
    """Render comprehensive asset security scan results."""
    st.subheader("🔍 Comprehensive Asset Security Scan")
    
    results = st.session_state.full_asset_scan_results
    asset_mapping = results.get('asset_security_mapping', {})
    
    # Summary metrics with asset context
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        asset_inventory = results.get('asset_inventory', {})
        total_assets = asset_inventory.get('total_assets', 0)
        st.metric("Assets Scanned", total_assets)
    
    with col2:
        security_response = results.get('security_score', {})
        if security_response.get('success'):
            score = security_response.get('data', {}).get('overall_score', 0)
            st.metric("Security Score", f"{score}/100")
        else:
            st.metric("Security Score", "Error")
    
    with col3:
        recommendations = results.get('recommendations', {})
        if recommendations.get('success'):
            rec_count = len(recommendations.get('recommendations', []))
            st.metric("Asset Recommendations", rec_count)
        else:
            st.metric("Asset Recommendations", "Error")
    
    with col4:
        compliance_response = results.get('compliance', {})
        if compliance_response.get('success'):
            compliant = compliance_response.get('data', {}).get('compliant', False)
            st.metric("Compliance", "✅" if compliant else "❌")
        else:
            st.metric("Compliance", "Error")
    
    # Asset security matrix
    st.subheader("🎯 Asset Security Matrix")
    matrix = asset_mapping.get('asset_security_matrix', {})
    if matrix:
        matrix_data = []
        for asset_type, details in matrix.items():
            matrix_data.append({
                "Asset Type": asset_type,
                "Count": details.get('count', 0),
                "Priority": details.get('security_priority', 'Unknown'),
                "Status": details.get('monitoring_status', 'Unknown'),
                "Last Scan": details.get('last_scan', 'Unknown')
            })
        
        df = pd.DataFrame(matrix_data)
        st.dataframe(df, use_container_width=True)
    
    # Detailed results in tabs with asset context
    tab1, tab2, tab3, tab4 = st.tabs(["Asset Security", "API Analysis", "Recommendations", "Compliance"])
    
    with tab1:
        if results.get('asset_inventory', {}).get('success'):
            render_asset_security_detailed(results['asset_inventory'])
        else:
            st.error("Failed to get asset inventory data")
    
    with tab2:
        if results.get('enabled_apis', {}).get('success'):
            render_api_analysis_with_assets(results['enabled_apis'], results['asset_inventory'])
        else:
            st.error("Failed to get API data")
    
    with tab3:
        if results.get('recommendations', {}).get('success'):
            render_asset_aware_recommendations(results['recommendations'], results['asset_inventory'])
        else:
            st.error("Failed to get recommendations")
    
    with tab4:
        if results.get('compliance', {}).get('success'):
            render_asset_compliance_analysis(results['compliance'], results['asset_inventory'])
        else:
            st.error("Failed to get compliance data")


def render_asset_security_detailed(asset_data: Dict):
    """Render detailed asset security information."""
    st.subheader("📋 Asset Security Details")
    
    categories = asset_data.get('asset_categories', {})
    if categories:
        for category, count in categories.items():
            with st.expander(f"🎯 {category} ({count} assets)"):
                st.metric("Asset Count", count)
                st.markdown("**Security Considerations:**")
                
                # Category-specific security info
                if "compute" in category.lower():
                    st.markdown("• Monitor for unauthorized access")
                    st.markdown("• Ensure proper patch management")
                    st.markdown("• Review firewall rules")
                elif "storage" in category.lower():
                    st.markdown("• Check data encryption status")
                    st.markdown("• Review access permissions")
                    st.markdown("• Monitor for data exfiltration")
                elif "network" in category.lower():
                    st.markdown("• Audit network security groups")
                    st.markdown("• Monitor traffic patterns")
                    st.markdown("• Check VPN configurations")
                else:
                    st.markdown("• Review resource permissions")
                    st.markdown("• Monitor usage patterns")
                    st.markdown("• Check compliance status")


def render_api_analysis_with_assets(api_data: Dict, asset_data: Dict):
    """Render API analysis with asset correlation."""
    st.subheader("🔗 API-Asset Correlation")
    
    enabled_apis = api_data.get('data', {}).get('enabled_apis', [])
    total_assets = asset_data.get('total_assets', 0)
    
    if enabled_apis and total_assets > 0:
        st.metric("API-to-Asset Ratio", f"{len(enabled_apis)}/{total_assets}")
        
        # Show correlation insights
        st.markdown("**Correlation Insights:**")
        st.markdown(f"• {len(enabled_apis)} APIs enabled for {total_assets} assets")
        st.markdown(f"• Average {len(enabled_apis)/max(total_assets, 1):.1f} APIs per asset")
        
        # API risk analysis
        high_risk_apis = [api for api in enabled_apis if api.get('risk_level') == 'high']
        if high_risk_apis:
            st.warning(f"⚠️ {len(high_risk_apis)} high-risk APIs detected with access to {total_assets} assets")


def render_asset_aware_recommendations(rec_data: Dict, asset_data: Dict):
    """Render recommendations with asset awareness."""
    st.subheader("🎯 Asset-Driven Recommendations")
    
    recommendations = rec_data.get('recommendations', [])
    total_assets = asset_data.get('total_assets', 0)
    
    if recommendations:
        for rec in recommendations[:5]:  # Top 5
            priority = rec.get('priority', 'medium')
            priority_emoji = {"high": "🔴", "medium": "🟡", "low": "🟢"}.get(priority, "⚪")
            
            with st.expander(f"{priority_emoji} {rec.get('title', 'Recommendation')}"):
                st.markdown(f"**Priority:** {priority.title()}")
                st.markdown(f"**Asset Impact:** Affects {total_assets} assets")
                st.markdown(f"**Description:** {rec.get('description', 'No description')}")
                
                # Asset-specific context
                categories = asset_data.get('asset_categories', {})
                if categories:
                    st.markdown("**Relevant Asset Types:**")
                    for category in list(categories.keys())[:3]:  # Top 3 categories
                        st.markdown(f"• {category}")


def render_asset_compliance_analysis(compliance_data: Dict, asset_data: Dict):
    """Render compliance analysis with asset correlation."""
    st.subheader("📋 Asset Compliance Analysis")
    
    compliant = compliance_data.get('data', {}).get('compliant', False)
    total_assets = asset_data.get('total_assets', 0)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Compliance Status", "✅ Compliant" if compliant else "❌ Non-Compliant")
    
    with col2:
        st.metric("Assets Under Compliance", total_assets)
    
    # Asset-specific compliance insights
    categories = asset_data.get('asset_categories', {})
    if categories:
        st.subheader("📊 Compliance by Asset Type")
        for category, count in categories.items():
            compliance_score = 85 if compliant else 65  # Simulate category compliance
            score_color = "success" if compliance_score >= 80 else "warning"
            
            st.markdown(f"**{category}** ({count} assets)")
            getattr(st, score_color)(f"Compliance Score: {compliance_score}%")


def render_security_summary_card():
    """Render enhanced security summary card with asset integration."""
    with st.container():
        st.subheader("🛡️ Asset Security Status")
        
        # Get asset-aware security data
        project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
        asset_service = AssetDataService()
        
        try:
            asset_data = asset_service.get_asset_summary(project_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if asset_data.get('success'):
                    security_score = asset_data.get('security_score', 0)
                    st.metric(
                        "Asset Security Score",
                        f"{security_score}/100",
                        help=f"Based on {asset_data.get('total_assets', 0)} assets"
                    )
                else:
                    st.metric("Security Score", "Scan Required")
            
            with col2:
                if asset_data.get('success'):
                    high_risk = asset_data.get('high_risk_count', 0)
                    delta_color = "inverse" if high_risk > 0 else "normal"
                    st.metric(
                        "High-Risk Assets",
                        high_risk,
                        delta_color=delta_color,
                        help="Assets requiring immediate security attention"
                    )
                else:
                    st.metric("Risk Assets", "Analyzing...")
            
        except Exception as e:
            st.error(f"Failed to load asset security data: {str(e)[:50]}...")
        
        if st.button("Asset Security Analysis", key="asset_security_scan"):
            st.session_state.page = "security"
            st.rerun()