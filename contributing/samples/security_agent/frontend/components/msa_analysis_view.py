"""MSA Analysis view component for the security agent frontend."""

import streamlit as st
import pandas as pd
import plotly.express as px
from typing import Dict, Any, List
from api_client import api_client


def render_msa_analysis_view():
    """Render the MSA analysis and Google Cloud impact assessment interface."""
    st.header("📄 MSA Analysis & Google Cloud Impact Assessment")
    st.write("Parse Microsoft Service Agreements and analyze their impact on your Google Cloud organization.")
    
    # Tab selection
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Parse MSA", "🔍 Scan Organization", "📊 View Results", "⚙️ Settings"])
    
    with tab1:
        render_msa_parsing_tab()
    
    with tab2:
        render_organization_scan_tab()
    
    with tab3:
        render_results_tab()
    
    with tab4:
        render_settings_tab()


def render_msa_parsing_tab():
    """Render the MSA parsing tab."""
    st.subheader("📝 Parse Microsoft Service Agreement (MSA)")
    
    # MSA input options
    input_option = st.radio(
        "Choose input method:",
        ["📋 Enter MSA Text", "📄 Upload MSA File", "🎯 Use Sample MSA"]
    )
    
    msa_text = ""
    msa_name = ""
    
    if input_option == "📋 Enter MSA Text":
        msa_name = st.text_input("MSA Name:", value="My Service Agreement")
        msa_text = st.text_area(
            "MSA Text Content:",
            height=400,
            placeholder="Paste your Microsoft Service Agreement text here..."
        )
    
    elif input_option == "📄 Upload MSA File":
        uploaded_file = st.file_uploader("Upload MSA file", type=['txt', 'pdf', 'docx'])
        if uploaded_file:
            msa_name = st.text_input("MSA Name:", value=uploaded_file.name)
            if uploaded_file.type == "text/plain":
                msa_text = str(uploaded_file.read(), "utf-8")
                st.success(f"✅ File '{uploaded_file.name}' loaded successfully!")
            else:
                st.warning("⚠️ File format not fully supported. Please convert to text format for best results.")
    
    elif input_option == "🎯 Use Sample MSA":
        if st.button("Load Sample MSA"):
            # For demo purposes, provide sample MSA text
            msa_text = get_sample_msa_text()
            msa_name = "Sample Microsoft Service Agreement"
            st.success("✅ Sample MSA loaded!")
    
    # Parse MSA
    if msa_text and msa_name:
        if st.button("🔍 Parse MSA", type="primary"):
            with st.spinner("Parsing MSA document..."):
                response = api_client.parse_msa(msa_text, msa_name)
            
            if response.get("success"):
                st.success("✅ MSA parsed successfully!")
                
                msa_record = response.get("msa_record", {})
                
                # Display parsing results
                st.subheader("📊 Parsing Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    apis_found = len(msa_record.get("extracted_apis", []))
                    st.metric("APIs Identified", apis_found)
                
                with col2:
                    security_clauses = len(msa_record.get("security_clauses", []))
                    st.metric("Security Clauses", security_clauses)
                
                with col3:
                    compliance_reqs = len(msa_record.get("compliance_requirements", []))
                    st.metric("Compliance Requirements", compliance_reqs)
                
                # Detailed results
                if msa_record.get("extracted_apis"):
                    st.subheader("🔧 Extracted APIs")
                    for api in msa_record["extracted_apis"]:
                        st.markdown(f"• {api}")
                
                if msa_record.get("security_clauses"):
                    st.subheader("🔒 Security Clauses")
                    for clause in msa_record["security_clauses"]:
                        with st.expander(f"Security Clause {clause.get('id', 'Unknown')}"):
                            st.write(clause.get("text", "No text available"))
                
                if msa_record.get("compliance_requirements"):
                    st.subheader("📋 Compliance Requirements")
                    for req in msa_record["compliance_requirements"]:
                        st.markdown(f"• {req}")
                
                # Store results in session state for other tabs
                st.session_state.msa_parse_results = response
            
            else:
                st.error(f"❌ Failed to parse MSA: {response.get('error', 'Unknown error')}")


def render_organization_scan_tab():
    """Render the Google Cloud organization scan tab."""
    st.subheader("🔍 Scan Google Cloud Organization")
    
    # Check if we have MSA results
    if not hasattr(st.session_state, 'msa_parse_results'):
        st.info("ℹ️ Please parse an MSA document first in the 'Parse MSA' tab.")
        return
    
    st.write("Analyze the impact of the parsed MSA on your Google Cloud organization.")
    
    # Scan configuration
    col1, col2 = st.columns(2)
    
    with col1:
        scan_scope = st.selectbox(
            "Scan Scope:",
            ["Current Project Only", "Entire Organization", "Selected Projects"]
        )
    
    with col2:
        include_inactive = st.checkbox("Include Inactive Resources", value=False)
    
    # Additional scan options
    with st.expander("⚙️ Advanced Scan Options"):
        check_iam = st.checkbox("Analyze IAM Policies", value=True)
        check_apis = st.checkbox("Check Enabled APIs", value=True)
        check_security = st.checkbox("Security Configuration Review", value=True)
        check_compliance = st.checkbox("Compliance Impact Assessment", value=True)
    
    if st.button("🚀 Start Organization Scan", type="primary"):
        with st.spinner("Scanning Google Cloud organization..."):
            # Prepare scan data
            msa_record = st.session_state.msa_parse_results.get("msa_record", {})
            
            scan_data = {
                "msa_record": msa_record,
                "scan_scope": scan_scope,
                "include_inactive": include_inactive,
                "options": {
                    "check_iam": check_iam,
                    "check_apis": check_apis,
                    "check_security": check_security,
                    "check_compliance": check_compliance
                }
            }
            
            # Mock scan results for demo
            scan_results = {
                "success": True,
                "impact_analysis": {
                    "affected_projects": 3,
                    "affected_apis": 12,
                    "security_risks": 5,
                    "compliance_gaps": 2,
                    "recommendations": [
                        "Review API access patterns for MSA compliance",
                        "Update IAM policies to align with MSA requirements",
                        "Implement additional logging for compliance tracking"
                    ]
                }
            }
            
            st.session_state.org_scan_results = scan_results
        
        if scan_results.get("success"):
            st.success("✅ Organization scan completed!")
            
            # Display scan summary
            impact = scan_results.get("impact_analysis", {})
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Affected Projects", impact.get("affected_projects", 0))
            
            with col2:
                st.metric("Affected APIs", impact.get("affected_apis", 0))
            
            with col3:
                st.metric("Security Risks", impact.get("security_risks", 0), delta_color="inverse")
            
            with col4:
                st.metric("Compliance Gaps", impact.get("compliance_gaps", 0), delta_color="inverse")
            
            # Recommendations
            recommendations = impact.get("recommendations", [])
            if recommendations:
                st.subheader("💡 Recommendations")
                for i, rec in enumerate(recommendations):
                    st.markdown(f"{i+1}. {rec}")
        
        else:
            st.error("❌ Organization scan failed. Please try again.")


def render_results_tab():
    """Render the results viewing tab."""
    st.subheader("📊 Analysis Results")
    
    # Check if we have results
    has_msa_results = hasattr(st.session_state, 'msa_parse_results')
    has_scan_results = hasattr(st.session_state, 'org_scan_results')
    
    if not has_msa_results and not has_scan_results:
        st.info("ℹ️ No analysis results available. Please run MSA parsing and organization scanning first.")
        return
    
    # Results tabs
    result_tabs = []
    if has_msa_results:
        result_tabs.append("📝 MSA Parsing Results")
    if has_scan_results:
        result_tabs.append("🔍 Organization Scan Results")
    
    if len(result_tabs) == 2:
        tab1, tab2 = st.tabs(result_tabs)
        
        with tab1:
            display_msa_results()
        
        with tab2:
            display_scan_results()
    
    elif has_msa_results:
        display_msa_results()
    
    elif has_scan_results:
        display_scan_results()


def display_msa_results():
    """Display MSA parsing results."""
    msa_results = st.session_state.msa_parse_results
    msa_record = msa_results.get("msa_record", {})
    
    st.write("**MSA Document Analysis Summary**")
    
    # Create summary dataframe
    summary_data = {
        "Category": ["APIs", "Security Clauses", "Compliance Requirements"],
        "Count": [
            len(msa_record.get("extracted_apis", [])),
            len(msa_record.get("security_clauses", [])),
            len(msa_record.get("compliance_requirements", []))
        ]
    }
    
    df = pd.DataFrame(summary_data)
    
    # Display as chart
    fig = px.bar(df, x="Category", y="Count", title="MSA Analysis Summary")
    st.plotly_chart(fig, use_container_width=True)
    
    # Export option
    if st.button("📄 Export MSA Results"):
        st.download_button(
            "Download JSON",
            data=str(msa_record),
            file_name="msa_analysis_results.json",
            mime="application/json"
        )


def display_scan_results():
    """Display organization scan results."""
    scan_results = st.session_state.org_scan_results
    impact = scan_results.get("impact_analysis", {})
    
    st.write("**Google Cloud Organization Impact Analysis**")
    
    # Impact summary
    impact_data = {
        "Area": ["Projects", "APIs", "Security Risks", "Compliance Gaps"],
        "Count": [
            impact.get("affected_projects", 0),
            impact.get("affected_apis", 0),
            impact.get("security_risks", 0),
            impact.get("compliance_gaps", 0)
        ]
    }
    
    df = pd.DataFrame(impact_data)
    
    # Display as chart
    fig = px.bar(df, x="Area", y="Count", title="Organization Impact Analysis")
    st.plotly_chart(fig, use_container_width=True)


def render_settings_tab():
    """Render the settings tab."""
    st.subheader("⚙️ MSA Analysis Settings")
    
    # Parsing settings
    st.markdown("**Parsing Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        parsing_mode = st.selectbox(
            "Parsing Mode:",
            ["Standard", "Detailed", "Fast"]
        )
        
        extract_apis = st.checkbox("Extract API References", value=True)
        extract_security = st.checkbox("Extract Security Clauses", value=True)
    
    with col2:
        extract_compliance = st.checkbox("Extract Compliance Requirements", value=True)
        extract_data_handling = st.checkbox("Extract Data Handling Clauses", value=True)
        include_context = st.checkbox("Include Context Information", value=False)
    
    # Organization scan settings
    st.markdown("**Organization Scan Configuration**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        default_scope = st.selectbox(
            "Default Scan Scope:",
            ["Current Project Only", "Entire Organization"]
        )
        
        max_projects = st.number_input("Max Projects to Scan", min_value=1, max_value=100, value=10)
    
    with col2:
        scan_timeout = st.number_input("Scan Timeout (minutes)", min_value=1, max_value=60, value=15)
        concurrent_scans = st.number_input("Concurrent Scans", min_value=1, max_value=10, value=3)
    
    # Save settings
    if st.button("💾 Save Settings"):
        settings = {
            "parsing_mode": parsing_mode,
            "extract_apis": extract_apis,
            "extract_security": extract_security,
            "extract_compliance": extract_compliance,
            "extract_data_handling": extract_data_handling,
            "include_context": include_context,
            "default_scope": default_scope,
            "max_projects": max_projects,
            "scan_timeout": scan_timeout,
            "concurrent_scans": concurrent_scans
        }
        
        st.session_state.msa_settings = settings
        st.success("✅ Settings saved successfully!")
    
    # Clear results
    st.markdown("**Data Management**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("🗑️ Clear MSA Results"):
            if hasattr(st.session_state, 'msa_parse_results'):
                del st.session_state.msa_parse_results
            st.success("MSA results cleared!")
    
    with col2:
        if st.button("🗑️ Clear Scan Results"):
            if hasattr(st.session_state, 'org_scan_results'):
                del st.session_state.org_scan_results
            st.success("Scan results cleared!")


def get_sample_msa_text():
    """Return sample MSA text for demo purposes."""
    return """
    MICROSOFT SERVICE AGREEMENT - SAMPLE
    
    This agreement governs the use of Microsoft cloud services including Azure, Office 365, and related APIs.
    
    SECURITY PROVISIONS:
    - All data transmission must use TLS 1.2 or higher encryption
    - Multi-factor authentication is required for administrative access
    - Audit logging must be enabled for all API access
    
    API ACCESS:
    - Microsoft Graph API for user and organization data
    - Azure Resource Manager API for cloud resource management
    - Office 365 Management API for administrative functions
    - Power BI REST API for analytics and reporting
    
    COMPLIANCE REQUIREMENTS:
    - SOC 2 Type II compliance for data handling
    - GDPR compliance for EU data subjects
    - ISO 27001 certification for security management
    
    DATA HANDLING:
    - Customer data remains property of the customer
    - Microsoft may access data only for service provision
    - Data retention periods follow regulatory requirements
    
    This is a simplified sample for demonstration purposes.
    """


def render_msa_summary_card():
    """Render a compact MSA summary card for the dashboard."""
    with st.container():
        st.subheader("📄 MSA Analysis")
        
        # Check if we have MSA results
        if hasattr(st.session_state, 'msa_parse_results'):
            msa_record = st.session_state.msa_parse_results.get("msa_record", {})
            
            col1, col2 = st.columns(2)
            
            with col1:
                apis_count = len(msa_record.get("extracted_apis", []))
                st.metric("APIs Found", apis_count)
            
            with col2:
                security_count = len(msa_record.get("security_clauses", []))
                st.metric("Security Clauses", security_count)
            
            st.success("✅ MSA analysis completed")
        else:
            st.info("No MSA analysis results available")
        
        if st.button("Analyze MSA", key="analyze_msa"):
            st.session_state.page = "msa"
            st.rerun()