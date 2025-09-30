"""
Service Discovery & On-Demand Analysis Page
Allows users to dynamically analyze any GCP service
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
from services.adk_service import ADKService
from services.metrics_service import MetricsService
from components.chat_widget import create_chat_widget
from components.charts import SecurityCharts
import time

# Configure page
st.set_page_config(
    page_title="Service Discovery & Analysis",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize services
adk_service = ADKService()
metrics_service = MetricsService()

# Page header
st.title("🔍 Service Discovery & On-Demand Analysis")
st.markdown("Dynamically discover and analyze any GCP service in real-time")

# Create tabs
tabs = st.tabs(["📊 Service Discovery", "🔎 On-Demand Analysis", "📈 Service Metrics", "🤖 AI Analysis"])

with tabs[0]:
    st.header("GCP Service Discovery")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Service discovery controls
        st.subheader("Discover Services")

        discovery_col1, discovery_col2, discovery_col3 = st.columns(3)

        with discovery_col1:
            if st.button("🔄 Discover All Services", type="primary", use_container_width=True):
                with st.spinner("Discovering enabled GCP services..."):
                    try:
                        # Call ADK agent to discover services
                        response = adk_service.send_message("Discover all enabled GCP services in the project")

                        if response and response.get("success"):
                            st.success("✅ Service discovery complete!")

                            # Store discovered services in session state
                            if "services" in response:
                                st.session_state.discovered_services = response["services"]
                        else:
                            st.error("Failed to discover services")
                    except Exception as e:
                        st.error(f"Error discovering services: {str(e)}")

        with discovery_col2:
            if st.button("🔍 Check Specific Service", use_container_width=True):
                st.session_state.show_service_check = True

        with discovery_col3:
            if st.button("📋 Export Service List", use_container_width=True):
                if hasattr(st.session_state, 'discovered_services'):
                    services_df = pd.DataFrame(st.session_state.discovered_services)
                    csv = services_df.to_csv(index=False)
                    st.download_button(
                        label="Download CSV",
                        data=csv,
                        file_name=f"gcp_services_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )

    with col2:
        # Service statistics
        st.subheader("Service Statistics")
        if hasattr(st.session_state, 'discovered_services'):
            total_services = len(st.session_state.discovered_services)
            st.metric("Total Services", total_services)
            st.metric("APIs Enabled", f"{total_services} APIs")
            st.metric("Last Updated", datetime.now().strftime("%H:%M:%S"))
        else:
            st.info("Click 'Discover All Services' to view statistics")

    # Display discovered services
    if hasattr(st.session_state, 'discovered_services') and st.session_state.discovered_services:
        st.subheader("Discovered Services")

        # Create a DataFrame for display
        services_data = []
        for service in st.session_state.discovered_services:
            services_data.append({
                "Service": service.get("name", "Unknown"),
                "API": service.get("api", ""),
                "Status": "✅ Enabled" if service.get("enabled") else "❌ Disabled",
                "Resource Types": ", ".join(service.get("resource_types", [])),
                "Description": service.get("description", "")
            })

        services_df = pd.DataFrame(services_data)

        # Add search filter
        search_term = st.text_input("🔍 Filter services", placeholder="Type to search...")
        if search_term:
            mask = services_df.apply(lambda x: x.astype(str).str.contains(search_term, case=False).any(), axis=1)
            filtered_df = services_df[mask]
        else:
            filtered_df = services_df

        # Display table with selection
        selected_service = st.dataframe(
            filtered_df,
            use_container_width=True,
            height=400,
            hide_index=True,
            selection_mode="single-row",
            on_select="rerun"
        )

        # If a service is selected, show analysis options
        if selected_service and selected_service.selection.rows:
            selected_idx = selected_service.selection.rows[0]
            selected_svc = filtered_df.iloc[selected_idx]
            st.session_state.selected_service = selected_svc["Service"]
            st.info(f"Selected: {selected_svc['Service']} - Navigate to 'On-Demand Analysis' tab to analyze")

with tabs[1]:
    st.header("On-Demand Service Analysis")

    # Service selection
    col1, col2 = st.columns([2, 1])

    with col1:
        # Service selector
        service_options = ["Select a service..."]
        if hasattr(st.session_state, 'discovered_services'):
            service_options.extend([s["name"] for s in st.session_state.discovered_services])

        # Pre-select if service was chosen in Discovery tab
        default_idx = 0
        if hasattr(st.session_state, 'selected_service'):
            if st.session_state.selected_service in service_options:
                default_idx = service_options.index(st.session_state.selected_service)

        selected_service = st.selectbox(
            "Select GCP Service to Analyze",
            service_options,
            index=default_idx,
            help="Choose a service or discover services in the Discovery tab first"
        )

    with col2:
        # Quick actions
        st.subheader("Quick Actions")
        analyze_btn = st.button("🔎 Analyze Service", type="primary", use_container_width=True)
        get_resources_btn = st.button("📦 Get Resources", use_container_width=True)

    # Analysis type selection
    st.subheader("Analysis Type")
    analysis_col1, analysis_col2, analysis_col3, analysis_col4 = st.columns(4)

    with analysis_col1:
        analyze_security = st.checkbox("🔒 Security Analysis", value=True)
    with analysis_col2:
        analyze_compliance = st.checkbox("✅ Compliance Check", value=True)
    with analysis_col3:
        analyze_cost = st.checkbox("💰 Cost Analysis", value=False)
    with analysis_col4:
        analyze_performance = st.checkbox("⚡ Performance Metrics", value=False)

    # Custom query section
    with st.expander("🔧 Advanced: Custom SQL Query", expanded=False):
        st.markdown("Write custom SQL queries for any service data")
        custom_query = st.text_area(
            "SQL Query",
            placeholder="SELECT * FROM `project.dataset.table` WHERE ...",
            height=100
        )

        col1, col2 = st.columns(2)
        with col1:
            run_custom_query = st.button("▶️ Run Custom Query", use_container_width=True)
        with col2:
            validate_query = st.button("✓ Validate Query", use_container_width=True)

    # Execute analysis
    if analyze_btn and selected_service != "Select a service...":
        with st.spinner(f"Analyzing {selected_service}..."):
            analysis_container = st.container()

            with analysis_container:
                # Build analysis request
                analysis_types = []
                if analyze_security:
                    analysis_types.append("security")
                if analyze_compliance:
                    analysis_types.append("compliance")
                if analyze_cost:
                    analysis_types.append("cost")
                if analyze_performance:
                    analysis_types.append("performance")

                # Call ADK agent for analysis
                query = f"Analyze {selected_service} service focusing on: {', '.join(analysis_types)}"
                response = adk_service.send_message(query)

                if response and response.get("success"):
                    st.success(f"✅ Analysis complete for {selected_service}")

                    # Display analysis results
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.subheader("Analysis Results")
                        st.markdown(response.get("message", "No results available"))

                    with col2:
                        st.subheader("Key Findings")
                        findings = response.get("findings", [])
                        if findings:
                            for finding in findings[:5]:
                                severity = finding.get("severity", "INFO")
                                icon = "🔴" if severity == "CRITICAL" else "🟡" if severity == "HIGH" else "🟢"
                                st.write(f"{icon} {finding.get('title', 'Finding')}")
                else:
                    st.error("Analysis failed. Please try again.")

    # Get resources
    if get_resources_btn and selected_service != "Select a service...":
        with st.spinner(f"Fetching resources for {selected_service}..."):
            query = f"Get all resources for {selected_service} service"
            response = adk_service.send_message(query)

            if response and response.get("success"):
                resources = response.get("resources", [])
                if resources:
                    st.subheader(f"Resources in {selected_service}")

                    # Create resources table
                    resources_df = pd.DataFrame(resources)
                    st.dataframe(resources_df, use_container_width=True, height=300)

                    # Resource statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Resources", len(resources))
                    with col2:
                        if "status" in resources_df.columns:
                            active = len(resources_df[resources_df["status"] == "ACTIVE"])
                            st.metric("Active Resources", active)
                    with col3:
                        if "region" in resources_df.columns:
                            regions = resources_df["region"].nunique()
                            st.metric("Regions", regions)
                else:
                    st.info("No resources found for this service")

    # Run custom query
    if run_custom_query and custom_query:
        with st.spinner("Executing custom query..."):
            response = adk_service.send_message(f"Run this SQL query: {custom_query}")

            if response and response.get("success"):
                st.success("Query executed successfully!")

                # Display results
                results = response.get("data", [])
                if results:
                    results_df = pd.DataFrame(results)
                    st.dataframe(results_df, use_container_width=True)

                    # Export option
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Results",
                        data=csv,
                        file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv"
                    )
            else:
                st.error("Query execution failed")

with tabs[2]:
    st.header("Service Metrics & Trends")

    # Service selection for metrics
    service_for_metrics = st.selectbox(
        "Select Service for Metrics",
        service_options if 'service_options' in locals() else ["Select a service..."],
        key="metrics_service_selector"
    )

    if service_for_metrics != "Select a service...":
        # Time range selector
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            time_range = st.selectbox("Time Range", ["Last 24 Hours", "Last 7 Days", "Last 30 Days", "Last 90 Days"])
        with col2:
            metric_type = st.selectbox("Metric Type", ["All Metrics", "Security", "Performance", "Cost", "Compliance"])
        with col3:
            if st.button("📊 Load Metrics", type="primary"):
                st.session_state.load_metrics = True

        if hasattr(st.session_state, 'load_metrics') and st.session_state.load_metrics:
            # Simulated metrics (in production, these would come from the backend)
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Resource Count",
                    "147",
                    "+12 (8.9%)",
                    help="Total resources in this service"
                )

            with col2:
                st.metric(
                    "Security Score",
                    "82%",
                    "+5%",
                    help="Overall security posture score"
                )

            with col3:
                st.metric(
                    "Monthly Cost",
                    "$12,453",
                    "-$1,234 (-9%)",
                    help="Estimated monthly cost"
                )

            with col4:
                st.metric(
                    "API Calls",
                    "1.2M",
                    "+150K",
                    help="API calls in selected period"
                )

            # Charts
            col1, col2 = st.columns(2)

            with col1:
                # Resource growth chart
                st.subheader("Resource Growth Trend")
                dates = pd.date_range(end=datetime.now(), periods=30)
                trend_data = pd.DataFrame({
                    'Date': dates,
                    'Resources': [100 + i * 2 + (i % 5) for i in range(30)]
                })
                fig = px.line(trend_data, x='Date', y='Resources', title="")
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                # Cost breakdown
                st.subheader("Cost Breakdown")
                cost_data = pd.DataFrame({
                    'Component': ['Compute', 'Storage', 'Network', 'Operations', 'Other'],
                    'Cost': [5000, 3500, 2000, 1500, 453]
                })
                fig = px.pie(cost_data, values='Cost', names='Component', title="")
                st.plotly_chart(fig, use_container_width=True)

            # Detailed metrics table
            st.subheader("Detailed Metrics")
            metrics_data = pd.DataFrame({
                'Metric': ['Availability', 'Latency (p99)', 'Error Rate', 'Throughput', 'CPU Usage', 'Memory Usage'],
                'Current': ['99.95%', '45ms', '0.02%', '1.2K req/s', '67%', '72%'],
                'Target': ['99.9%', '<50ms', '<0.1%', '>1K req/s', '<80%', '<85%'],
                'Status': ['✅', '✅', '✅', '✅', '✅', '✅']
            })
            st.dataframe(metrics_data, use_container_width=True, hide_index=True)

with tabs[3]:
    st.header("AI-Powered Analysis")

    # Analysis suggestions
    st.subheader("🤖 Intelligent Analysis Assistant")
    st.markdown("Get AI-powered recommendations for service analysis")

    col1, col2 = st.columns([2, 1])

    with col1:
        # Query input
        analysis_query = st.text_area(
            "What would you like to analyze?",
            placeholder="Example: Show me all Cloud Run services with high latency in the last 24 hours\n"
                       "Example: Find Compute Engine instances without backup policies\n"
                       "Example: Analyze BigQuery dataset costs by department",
            height=100
        )

    with col2:
        st.subheader("Quick Analysis")
        if st.button("🔍 Security Vulnerabilities", use_container_width=True):
            analysis_query = "Find security vulnerabilities across all services"
        if st.button("💰 Cost Optimization", use_container_width=True):
            analysis_query = "Identify cost optimization opportunities"
        if st.button("⚡ Performance Issues", use_container_width=True):
            analysis_query = "Detect performance bottlenecks in services"
        if st.button("✅ Compliance Gaps", use_container_width=True):
            analysis_query = "Check compliance gaps in all services"

    if st.button("🚀 Get AI Analysis", type="primary", use_container_width=True):
        if analysis_query:
            with st.spinner("AI is analyzing your request..."):
                # Get AI suggestions
                suggestion_query = f"Suggest analysis for: {analysis_query}"
                response = adk_service.send_message(suggestion_query)

                if response and response.get("success"):
                    st.success("Analysis complete!")

                    # Display AI recommendations
                    recommendations = response.get("recommendations", [])
                    if recommendations:
                        st.subheader("📋 Recommended Analyses")
                        for i, rec in enumerate(recommendations, 1):
                            with st.expander(f"{i}. {rec.get('title', 'Recommendation')}", expanded=(i==1)):
                                st.markdown(f"**Description:** {rec.get('description', '')}")
                                st.markdown(f"**Query:** `{rec.get('query', '')}`")

                                col1, col2 = st.columns(2)
                                with col1:
                                    if st.button(f"Run This Analysis", key=f"run_{i}"):
                                        st.session_state[f"run_analysis_{i}"] = rec.get('query')
                                with col2:
                                    st.markdown(f"**Priority:** {rec.get('priority', 'Medium')}")

                    # Display analysis results
                    st.subheader("📊 Analysis Results")
                    st.markdown(response.get("message", ""))
                else:
                    st.error("Analysis failed. Please try again.")
        else:
            st.warning("Please enter an analysis query")

    # Chat widget for interactive analysis
    st.divider()
    st.subheader("💬 Interactive Analysis Chat")
    st.markdown("Ask questions about any GCP service or get help with analysis")
    create_chat_widget(
        context="service_discovery",
        height=400,
        suggestions=[
            "What services are running in my project?",
            "Analyze security posture of Cloud Run services",
            "Show me unused resources across all services",
            "Which services have the highest costs?",
            "Find all publicly accessible resources"
        ]
    )

# Sidebar information
with st.sidebar:
    st.header("ℹ️ About Service Discovery")
    st.markdown("""
    This page provides on-demand analysis capabilities for any GCP service, not limited to pre-populated lists.

    **Features:**
    - 🔍 Dynamic service discovery
    - 📊 Real-time service analysis
    - 🔧 Custom SQL queries
    - 🤖 AI-powered recommendations
    - 📈 Service metrics & trends

    **Supported Services:**
    - Compute Engine
    - Cloud Storage
    - BigQuery
    - Cloud Run
    - Cloud Functions
    - Kubernetes Engine
    - Cloud SQL
    - Firestore
    - Pub/Sub
    - And 15+ more...

    **Tips:**
    - Use the Discovery tab to find all enabled services
    - Select a service for detailed analysis
    - Write custom queries for specific needs
    - Let AI suggest relevant analyses
    """)

    # Quick stats
    st.divider()
    st.subheader("📊 Quick Stats")
    if hasattr(st.session_state, 'discovered_services'):
        st.metric("Services Discovered", len(st.session_state.discovered_services))
        st.metric("Last Discovery", datetime.now().strftime("%H:%M"))
    else:
        st.info("No services discovered yet")