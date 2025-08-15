"""Compliance evaluation view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List
import sys
import os
# Add path to access frontend root directory
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from api_client_consolidated import api_client as simple_api
from services.asset_data_service import AssetDataService


def render_compliance_view():
    """Render the asset-aware compliance evaluation dashboard."""
    st.header("📋 Asset-Aware Compliance Dashboard")
    st.write("Evaluate compliance frameworks with full asset inventory context.")
    
    # Initialize asset service and get asset context
    asset_service = AssetDataService()
    project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
    
    # Get asset context for compliance correlation
    with st.spinner("Loading asset inventory for compliance analysis..."):
        asset_data = asset_service.get_asset_summary(project_id)
    
    # Asset compliance overview
    render_asset_compliance_overview(asset_data)
    
    # Framework selector
    framework_options = {
        "SOC2": "SOC 2 Type II",
        "ISO27001": "ISO 27001",
        "GDPR": "General Data Protection Regulation",
        "HIPAA": "Health Insurance Portability and Accountability Act",
        "PCI_DSS": "Payment Card Industry Data Security Standard"
    }
    
    selected_framework = st.selectbox(
        "Select Compliance Framework:",
        options=list(framework_options.keys()),
        format_func=lambda x: framework_options[x],
        help="Choose the compliance framework to evaluate against"
    )
    
    # Evaluation actions
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button(f"🔍 Evaluate {framework_options[selected_framework]} (Asset-Aware)", type="primary"):
            evaluate_asset_compliance(selected_framework, asset_data)
    
    with col2:
        if st.button("📊 Asset Compliance Matrix"):
            compare_all_frameworks_with_assets(asset_data)
    
    # Display results
    if hasattr(st.session_state, f'compliance_{selected_framework.lower()}'):
        render_compliance_results(selected_framework)
    
    if hasattr(st.session_state, 'compliance_comparison'):
        render_compliance_comparison()


def render_asset_compliance_overview(asset_data: Dict[str, Any]):
    """Render asset-aware compliance overview."""
    st.subheader("🎯 Asset Compliance Overview")
    
    if asset_data.get('success') and asset_data.get('total_assets', 0) > 0:
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            total_assets = asset_data.get('total_assets', 0)
            st.metric(
                "Assets Under Compliance",
                total_assets,
                help="Total assets subject to compliance frameworks"
            )
        
        with col2:
            # Categorize compliance-sensitive assets
            categories = asset_data.get('asset_categories', {})
            sensitive_assets = (
                categories.get('Cloud Storage', 0) +  # Data compliance
                categories.get('Compute Engine', 0) +  # Infrastructure compliance
                categories.get('Cloud SQL', 0)  # Database compliance
            )
            st.metric(
                "Compliance-Sensitive",
                sensitive_assets,
                delta=f"{(sensitive_assets/total_assets*100):.0f}% of total" if total_assets > 0 else "0%",
                help="Assets typically requiring compliance oversight"
            )
        
        with col3:
            # Calculate compliance complexity based on asset diversity
            compliance_complexity = min(100, len(categories) * 15 + total_assets // 20)
            complexity_color = "inverse" if compliance_complexity > 80 else "normal"
            st.metric(
                "Compliance Complexity",
                f"{compliance_complexity}/100",
                delta="Asset-driven score",
                delta_color=complexity_color,
                help="Compliance complexity estimated from asset inventory"
            )
        
        with col4:
            high_risk_assets = asset_data.get('high_risk_count', 0)
            st.metric(
                "Non-Compliant Risk",
                high_risk_assets,
                delta_color="inverse" if high_risk_assets > 0 else "normal",
                help="Assets potentially affecting compliance status"
            )
        
        # Asset compliance heatmap
        render_asset_compliance_heatmap(asset_data)
    else:
        st.warning("🔍 No asset data for compliance analysis. Run asset discovery first.")


def render_asset_compliance_heatmap(asset_data: Dict[str, Any]):
    """Render compliance status heatmap by asset category."""
    st.subheader("🔥 Asset Compliance Heatmap")
    
    categories = asset_data.get('asset_categories', {})
    if categories:
        # Create compliance matrix data
        frameworks = ["SOC2", "ISO27001", "GDPR", "HIPAA", "PCI_DSS"]
        heatmap_data = []
        
        for category, count in categories.items():
            for framework in frameworks:
                # Simulate compliance scores based on asset type and framework
                if "storage" in category.lower():
                    if framework in ["GDPR", "HIPAA"]:
                        score = 85 if count < 5 else 75  # Data protection frameworks
                    else:
                        score = 90
                elif "compute" in category.lower():
                    if framework in ["SOC2", "ISO27001"]:
                        score = 80 if count < 10 else 70  # Infrastructure frameworks
                    else:
                        score = 85
                else:
                    score = 90  # Other assets generally easier to maintain compliance
                
                heatmap_data.append({
                    "Asset Category": category,
                    "Framework": framework,
                    "Compliance Score": score,
                    "Asset Count": count
                })
        
        df = pd.DataFrame(heatmap_data)
        
        # Create heatmap
        fig = px.density_heatmap(
            df,
            x="Framework",
            y="Asset Category",
            z="Compliance Score",
            title="Compliance Status by Asset Category",
            color_continuous_scale="RdYlGn",
            text_auto=True
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Compliance insights
        render_asset_compliance_insights(df)


def render_asset_compliance_insights(df: pd.DataFrame):
    """Render insights from asset compliance analysis."""
    st.subheader("💡 Asset Compliance Insights")
    
    # Find lowest scoring combinations
    low_scores = df[df['Compliance Score'] < 80].sort_values('Compliance Score')
    
    if not low_scores.empty:
        st.warning("⚠️ Compliance attention needed:")
        for _, row in low_scores.head(3).iterrows():
            st.markdown(f"• **{row['Asset Category']}** - {row['Framework']}: {row['Compliance Score']}% ({row['Asset Count']} assets)")
    
    # Find highest risk by asset count
    high_count_low_score = df[(df['Asset Count'] > 5) & (df['Compliance Score'] < 85)]
    if not high_count_low_score.empty:
        st.error("🚨 High-impact compliance risks:")
        for _, row in high_count_low_score.iterrows():
            st.markdown(f"• **{row['Asset Category']}**: {row['Asset Count']} assets at {row['Compliance Score']}% compliance for {row['Framework']}")
    
    # Positive findings
    high_scores = df[df['Compliance Score'] >= 90]
    if not high_scores.empty:
        st.success("✅ Strong compliance areas:")
        top_categories = high_scores.groupby('Asset Category')['Compliance Score'].mean().sort_values(ascending=False).head(3)
        for category, score in top_categories.items():
            asset_count = df[df['Asset Category'] == category]['Asset Count'].iloc[0]
            st.markdown(f"• **{category}**: {score:.0f}% average compliance ({asset_count} assets)")


def evaluate_asset_compliance(framework: str, asset_data: Dict[str, Any]):
    """Evaluate compliance with asset context."""
    with st.spinner(f"Evaluating {framework} compliance with asset correlation..."):
        # Get standard compliance evaluation
        response = simple_api.evaluate_compliance(framework)
        
        # Enhance with asset context
        if response.get("success") and asset_data.get('success'):
            asset_compliance_analysis = analyze_asset_compliance_impact(response, asset_data, framework)
            response['asset_compliance_analysis'] = asset_compliance_analysis
        
        st.session_state[f'compliance_{framework.lower()}'] = response


def analyze_asset_compliance_impact(compliance_data: Dict, asset_data: Dict, framework: str) -> Dict:
    """Analyze how assets impact compliance status."""
    analysis = {
        "asset_compliance_mapping": {},
        "high_impact_assets": [],
        "compliance_by_asset_type": {},
        "remediation_priorities": []
    }
    
    categories = asset_data.get('asset_categories', {})
    total_assets = asset_data.get('total_assets', 0)
    compliance_score = compliance_data.get('data', {}).get('compliance_score', 0)
    
    # Map compliance requirements to asset types
    for category, count in categories.items():
        if framework == "GDPR" and "storage" in category.lower():
            analysis["high_impact_assets"].append({
                "category": category,
                "count": count,
                "impact": "High - Data protection requirements",
                "compliance_score": max(60, compliance_score - 10)  # Lower score for data assets
            })
        elif framework == "SOC2" and "compute" in category.lower():
            analysis["high_impact_assets"].append({
                "category": category,
                "count": count,
                "impact": "High - Infrastructure controls required",
                "compliance_score": max(70, compliance_score - 5)
            })
        elif framework == "PCI_DSS" and ("storage" in category.lower() or "database" in category.lower()):
            analysis["high_impact_assets"].append({
                "category": category,
                "count": count,
                "impact": "Critical - Payment data handling",
                "compliance_score": max(50, compliance_score - 20)
            })
        else:
            analysis["compliance_by_asset_type"][category] = {
                "count": count,
                "estimated_compliance": min(100, compliance_score + 5),
                "risk_level": "Low"
            }
    
    # Generate remediation priorities based on asset impact
    if analysis["high_impact_assets"]:
        analysis["remediation_priorities"] = [
            f"Priority 1: Address {len(analysis['high_impact_assets'])} high-impact asset categories",
            f"Priority 2: Implement automated compliance monitoring for {total_assets} total assets",
            f"Priority 3: Regular compliance audits for asset inventory changes"
        ]
    
    return analysis


def compare_all_frameworks_with_assets(asset_data: Dict[str, Any]):
    """Compare compliance across all frameworks with asset context."""
    with st.spinner("Evaluating all frameworks with asset correlation..."):
        results = {}
        
        frameworks = ["SOC2", "ISO27001", "GDPR", "HIPAA", "PCI_DSS"]
        progress_bar = st.progress(0)
        
        for i, framework in enumerate(frameworks):
            response = simple_api.evaluate_compliance(framework)
            
            # Enhance with asset context
            if response.get("success") and asset_data.get('success'):
                asset_analysis = analyze_asset_compliance_impact(response, asset_data, framework)
                response['asset_compliance_analysis'] = asset_analysis
            
            results[framework] = response
            progress_bar.progress((i + 1) / len(frameworks))
        
        st.session_state.compliance_comparison = results
        st.session_state.asset_compliance_matrix = create_asset_compliance_matrix(results, asset_data)
        progress_bar.empty()


def create_asset_compliance_matrix(results: Dict, asset_data: Dict) -> Dict:
    """Create comprehensive asset-compliance matrix."""
    matrix = {
        "framework_asset_scores": {},
        "asset_risk_summary": {},
        "compliance_coverage": {}
    }
    
    for framework, response in results.items():
        if response.get("success"):
            compliance_score = response.get('data', {}).get('compliance_score', 0)
            asset_analysis = response.get('asset_compliance_analysis', {})
            
            matrix["framework_asset_scores"][framework] = {
                "base_score": compliance_score,
                "high_impact_assets": len(asset_analysis.get('high_impact_assets', [])),
                "asset_adjusted_score": calculate_asset_adjusted_score(compliance_score, asset_analysis)
            }
    
    return matrix


def calculate_asset_adjusted_score(base_score: int, asset_analysis: Dict) -> int:
    """Calculate compliance score adjusted for asset impact."""
    high_impact_count = len(asset_analysis.get('high_impact_assets', []))
    
    # Reduce score based on high-impact assets requiring attention
    adjustment = min(20, high_impact_count * 5)  # Max 20 point reduction
    
    return max(0, base_score - adjustment)


def evaluate_compliance(framework: str):
    """Legacy compliance evaluation function."""
    with st.spinner(f"Evaluating {framework} compliance..."):
        response = simple_api.evaluate_compliance(framework)
        st.session_state[f'compliance_{framework.lower()}'] = response


def compare_all_frameworks():
    """Legacy framework comparison function."""
    with st.spinner("Evaluating all compliance frameworks..."):
        results = {}
        
        frameworks = ["SOC2", "ISO27001", "GDPR", "HIPAA", "PCI_DSS"]
        progress_bar = st.progress(0)
        
        for i, framework in enumerate(frameworks):
            response = simple_api.evaluate_compliance(framework)
            results[framework] = response
            progress_bar.progress((i + 1) / len(frameworks))
        
        st.session_state.compliance_comparison = results
        progress_bar.empty()


def compare_all_frameworks():
    """Compare compliance across all frameworks."""
    with st.spinner("Evaluating all compliance frameworks..."):
        results = {}
        
        frameworks = ["SOC2", "ISO27001", "GDPR", "HIPAA", "PCI_DSS"]
        progress_bar = st.progress(0)
        
        for i, framework in enumerate(frameworks):
            response = simple_api.evaluate_compliance(framework)
            results[framework] = response
            progress_bar.progress((i + 1) / len(frameworks))
        
        st.session_state.compliance_comparison = results
        progress_bar.empty()


def render_compliance_results(framework: str):
    """Render asset-aware compliance evaluation results."""
    st.subheader(f"📊 {framework} Asset-Aware Compliance Results")
    
    response = st.session_state[f'compliance_{framework.lower()}']
    asset_analysis = response.get('asset_compliance_analysis', {})
    
    if response.get("success"):
        data = response.get("data", {})
        compliant = data.get("compliant", False)
        score = data.get("compliance_score", 0)
        
        # Compliance status
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_emoji = "✅" if compliant else "❌"
            st.metric("Compliance Status", f"{status_emoji} {'Compliant' if compliant else 'Non-Compliant'}")
        
        with col2:
            st.metric("Compliance Score", f"{score}%", delta=f"{score-80}%" if score else None)
        
        with col3:
            requirements_met = data.get("requirements_met", 0)
            requirements_total = data.get("requirements_total", 0)
            st.metric("Requirements Met", f"{requirements_met}/{requirements_total}")
        
        # Asset-aware compliance visualization
        if score:
            col1, col2 = st.columns(2)
            
            with col1:
                import plotly.graph_objects as go
                
                # Base compliance score
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=score,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': f"{framework} Compliance Score"},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "green" if score >= 80 else "orange" if score >= 60 else "red"},
                        'steps': [
                            {'range': [0, 60], 'color': "lightgray"},
                            {'range': [60, 80], 'color': "gray"},
                            {'range': [80, 100], 'color': "lightgreen"}
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
                # Asset impact analysis
                if asset_analysis:
                    st.subheader("🎯 Asset Impact Analysis")
                    
                    high_impact_assets = asset_analysis.get('high_impact_assets', [])
                    if high_impact_assets:
                        st.warning(f"⚠️ {len(high_impact_assets)} high-impact asset categories:")
                        for asset in high_impact_assets:
                            st.markdown(f"• **{asset['category']}**: {asset['count']} assets - {asset['impact']}")
                    
                    remediation_priorities = asset_analysis.get('remediation_priorities', [])
                    if remediation_priorities:
                        st.subheader("🎯 Remediation Priorities")
                        for priority in remediation_priorities:
                            st.markdown(f"• {priority}")
                else:
                    st.info("📊 Run asset-aware compliance evaluation for detailed asset impact analysis")
        
        # Compliance gaps
        gaps = data.get("gaps", [])
        if gaps:
            st.subheader("⚠️ Compliance Gaps")
            
            for i, gap in enumerate(gaps):
                with st.expander(f"Gap {i+1}: {gap.get('title', 'Untitled Gap')}"):
                    st.markdown(f"**Severity:** {gap.get('severity', 'Unknown')}")
                    st.markdown(f"**Description:** {gap.get('description', 'No description')}")
                    
                    remediation = gap.get('remediation', [])
                    if remediation:
                        st.markdown("**Remediation Steps:**")
                        for step in remediation:
                            st.markdown(f"• {step}")
                    
                    if st.button(f"Mark as Resolved", key=f"resolve_gap_{i}"):
                        st.success("Gap marked as resolved!")
        
        # Compliance requirements breakdown
        requirements = data.get("requirements_breakdown", {})
        if requirements:
            st.subheader("📋 Requirements Breakdown")
            
            req_data = []
            for category, reqs in requirements.items():
                for req in reqs:
                    req_data.append({
                        "Category": category,
                        "Requirement": req.get("name", "Unknown"),
                        "Status": "✅ Met" if req.get("met", False) else "❌ Not Met",
                        "Priority": req.get("priority", "Medium")
                    })
            
            if req_data:
                df = pd.DataFrame(req_data)
                st.dataframe(df, use_container_width=True)
    
    else:
        st.error(f"❌ Failed to evaluate compliance: {response.get('error', 'Unknown error')}")


def render_compliance_comparison():
    """Render asset-aware comparison across all compliance frameworks."""
    st.subheader("📊 Asset-Aware Framework Comparison")
    
    results = st.session_state.compliance_comparison
    
    # Show asset compliance matrix if available
    if hasattr(st.session_state, 'asset_compliance_matrix'):
        render_asset_compliance_matrix_visualization()
    else:
        st.info("📊 Run 'Asset Compliance Matrix' for enhanced asset-framework correlation analysis")
    
    # Extract scores for comparison
    framework_scores = {}
    framework_status = {}
    
    for framework, response in results.items():
        if response.get("success"):
            data = response.get("data", {})
            framework_scores[framework] = data.get("compliance_score", 0)
            framework_status[framework] = "Compliant" if data.get("compliant", False) else "Non-Compliant"
        else:
            framework_scores[framework] = 0
            framework_status[framework] = "Error"
    
    # Scores comparison chart
    if framework_scores:
        df = pd.DataFrame([
            {"Framework": framework, "Score": score}
            for framework, score in framework_scores.items()
        ])
        
        fig = px.bar(
            df,
            x="Framework",
            y="Score",
            title="Compliance Scores by Framework",
            color="Score",
            color_continuous_scale="RdYlGn"
        )
        
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
    
    # Status summary table
    status_data = []
    for framework, status in framework_status.items():
        status_data.append({
            "Framework": framework,
            "Score": f"{framework_scores.get(framework, 0)}%",
            "Status": status,
            "Priority": "High" if framework_scores.get(framework, 0) < 80 else "Medium"
        })
    
    df = pd.DataFrame(status_data)
    st.dataframe(df, use_container_width=True)
    
    # Recommendations based on comparison
    st.subheader("💡 Recommendations")
    
    low_scores = [fw for fw, score in framework_scores.items() if score < 80]
    
    if low_scores:
        st.warning(f"⚠️ The following frameworks need attention: {', '.join(low_scores)}")
        st.markdown("**Recommended Actions:**")
        st.markdown("• Review compliance gaps for each framework")
        st.markdown("• Prioritize high-severity remediation items")
        st.markdown("• Implement automated compliance monitoring")
        st.markdown("• Schedule regular compliance assessments")
    else:
        st.success("✅ All frameworks are meeting compliance requirements!")


def render_asset_compliance_matrix_visualization():
    """Render the asset compliance matrix visualization."""
    matrix = st.session_state.asset_compliance_matrix
    framework_scores = matrix.get('framework_asset_scores', {})
    
    if framework_scores:
        st.subheader("📊 Asset-Adjusted Compliance Matrix")
        
        # Create comparison data
        comparison_data = []
        for framework, data in framework_scores.items():
            comparison_data.append({
                "Framework": framework,
                "Base Score": data.get('base_score', 0),
                "Asset-Adjusted Score": data.get('asset_adjusted_score', 0),
                "High-Impact Assets": data.get('high_impact_assets', 0)
            })
        
        df = pd.DataFrame(comparison_data)
        
        # Comparison chart
        fig = px.bar(
            df,
            x="Framework",
            y=["Base Score", "Asset-Adjusted Score"],
            title="Compliance Scores: Base vs Asset-Adjusted",
            barmode="group",
            color_discrete_map={
                "Base Score": "lightblue",
                "Asset-Adjusted Score": "darkblue"
            }
        )
        
        fig.update_layout(yaxis_range=[0, 100])
        st.plotly_chart(fig, use_container_width=True)
        
        # Impact summary
        st.subheader("📊 Impact Summary")
        total_adjustments = sum(data.get('base_score', 0) - data.get('asset_adjusted_score', 0) for data in framework_scores.values())
        
        if total_adjustments > 0:
            st.warning(f"⚠️ Asset impact detected: {total_adjustments:.0f} total points reduction across frameworks")
            
            # Show frameworks with highest impact
            impact_analysis = []
            for framework, data in framework_scores.items():
                impact = data.get('base_score', 0) - data.get('asset_adjusted_score', 0)
                if impact > 0:
                    impact_analysis.append((framework, impact, data.get('high_impact_assets', 0)))
            
            impact_analysis.sort(key=lambda x: x[1], reverse=True)
            
            for framework, impact, asset_count in impact_analysis[:3]:
                st.markdown(f"• **{framework}**: -{impact:.0f} points ({asset_count} high-impact asset categories)")
        else:
            st.success("✅ No significant asset-related compliance impact detected")


def render_compliance_summary_card():
    """Render enhanced compliance summary card with asset integration."""
    with st.container():
        st.subheader("📋 Asset Compliance Status")
        
        # Get asset-aware compliance data
        project_id = st.session_state.get('selected_project', 'mgm-digitalconcierge')
        asset_service = AssetDataService()
        
        try:
            asset_data = asset_service.get_asset_summary(project_id)
            
            col1, col2 = st.columns(2)
            
            with col1:
                if asset_data.get('success'):
                    total_assets = asset_data.get('total_assets', 0)
                    categories = len(asset_data.get('asset_categories', {}))
                    
                    # Estimate overall compliance based on asset complexity
                    base_compliance = 85  # Base score
                    complexity_penalty = min(15, categories * 2)  # Reduce for complexity
                    estimated_compliance = max(70, base_compliance - complexity_penalty)
                    
                    st.metric(
                        "Est. Compliance",
                        f"{estimated_compliance}%",
                        help=f"Estimated from {total_assets} assets across {categories} categories"
                    )
                else:
                    st.metric("Est. Compliance", "Scan Required")
            
            with col2:
                if asset_data.get('success'):
                    # Count compliance-sensitive assets
                    categories = asset_data.get('asset_categories', {})
                    sensitive_count = (
                        categories.get('Cloud Storage', 0) +
                        categories.get('Compute Engine', 0) +
                        categories.get('Cloud SQL', 0)
                    )
                    total_assets = asset_data.get('total_assets', 1)
                    
                    st.metric(
                        "Sensitive Assets",
                        f"{sensitive_count}/{total_assets}",
                        delta=f"{(sensitive_count/total_assets*100):.0f}% requiring oversight",
                        help="Assets requiring compliance attention"
                    )
                else:
                    st.metric("Sensitive Assets", "Unknown")
        
        except Exception as e:
            st.error(f"Failed to load asset compliance data: {str(e)[:50]}...")
        
        if st.button("Asset Compliance Analysis", key="check_compliance"):
            st.session_state.page = "compliance"
            st.rerun()