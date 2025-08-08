"""Enhanced Multi-Agent Graph Visualization Component."""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from streamlit_agraph import agraph, Node, Edge, Config
import networkx as nx
import pandas as pd
import json
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import random
import math


def render_multi_agent_graph_view():
    """Render the enhanced multi-agent graph visualization interface."""
    st.header("🕸️ Multi-Agent System Graph")
    st.markdown("""
    **Advanced multi-agent system visualization with interactive dependency graphs, 
    risk propagation analysis, and real-time collaboration mapping.**
    """)
    
    # Sidebar controls
    with st.sidebar:
        st.subheader("🎛️ Graph Controls")
        
        graph_type = st.selectbox(
            "Graph Type",
            ["Service Dependencies", "Agent Collaboration", "Risk Propagation", "API Dependencies", "Multi-Agent Workflow"]
        )
        
        layout_algorithm = st.selectbox(
            "Layout Algorithm",
            ["Force-Directed", "Hierarchical", "Circular", "Grid", "Tree"]
        )
        
        show_labels = st.checkbox("Show Node Labels", value=True)
        show_metrics = st.checkbox("Show Performance Metrics", value=True)
        animate_data_flow = st.checkbox("Animate Data Flow", value=False)
        
        # Color scheme
        color_scheme = st.selectbox(
            "Color Scheme",
            ["Default", "Security Risk", "Performance", "Service Status", "Dark Theme"]
        )
    
    # Main graph area
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "🔗 Interactive Graph", 
        "📊 Graph Analytics", 
        "🎯 Agent Coordination", 
        "⚠️ Risk Analysis",
        "🔍 Graph Explorer"
    ])
    
    with tab1:
        render_interactive_graph(graph_type, layout_algorithm, show_labels, color_scheme, animate_data_flow)
    
    with tab2:
        render_graph_analytics()
    
    with tab3:
        render_agent_coordination()
        
    with tab4:
        render_risk_analysis()
        
    with tab5:
        render_graph_explorer()


def render_interactive_graph(graph_type: str, layout: str, show_labels: bool, color_scheme: str, animate: bool):
    """Render the main interactive graph visualization."""
    st.subheader(f"📈 {graph_type} Visualization")
    
    # Generate graph data based on type
    if graph_type == "Service Dependencies":
        graph_data = generate_service_dependency_graph()
    elif graph_type == "Agent Collaboration":
        graph_data = generate_agent_collaboration_graph()
    elif graph_type == "Risk Propagation":
        graph_data = generate_risk_propagation_graph()
    elif graph_type == "API Dependencies":
        graph_data = generate_api_dependency_graph()
    else:  # Multi-Agent Workflow
        graph_data = generate_multi_agent_workflow_graph()
    
    # Create visualization based on layout
    if layout == "Force-Directed":
        render_force_directed_graph(graph_data, show_labels, color_scheme, animate)
    elif layout == "Hierarchical":
        render_hierarchical_graph(graph_data, show_labels, color_scheme)
    elif layout == "Circular":
        render_circular_graph(graph_data, show_labels, color_scheme)
    elif layout == "Grid":
        render_grid_graph(graph_data, show_labels, color_scheme)
    else:  # Tree
        render_tree_graph(graph_data, show_labels, color_scheme)
    
    # Display graph statistics
    with st.expander("📊 Graph Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Nodes", len(graph_data["nodes"]))
        with col2:
            st.metric("Edges", len(graph_data["edges"]))
        with col3:
            st.metric("Components", calculate_connected_components(graph_data))
        with col4:
            st.metric("Max Degree", calculate_max_degree(graph_data))


def render_force_directed_graph(graph_data: Dict, show_labels: bool, color_scheme: str, animate: bool):
    """Render force-directed graph using streamlit-agraph."""
    
    # Convert to agraph format
    nodes = []
    edges = []
    
    # Color mapping based on scheme
    color_map = get_color_scheme(color_scheme)
    
    for node_data in graph_data["nodes"]:
        node_color = color_map.get(node_data.get("type", "default"), "#1f77b4")
        node_size = node_data.get("size", 20) * 2  # Scale for better visibility
        
        node = Node(
            id=node_data["id"], 
            label=node_data["label"] if show_labels else "",
            size=node_size,
            color=node_color,
            shape="dot" if node_data.get("type") != "agent" else "star",
            font={"size": 16 if show_labels else 0}
        )
        nodes.append(node)
    
    for edge_data in graph_data["edges"]:
        edge_color = color_map.get("edge", "#999999")
        edge_width = edge_data.get("weight", 1) * 2
        
        edge = Edge(
            source=edge_data["source"], 
            target=edge_data["target"],
            color=edge_color,
            width=edge_width,
            arrows={"to": {"enabled": True, "scaleFactor": 1}},
            smooth={"enabled": True, "type": "continuous"}
        )
        edges.append(edge)
    
    # Configure physics and interaction
    config = Config(
        width="100%",
        height=600,
        directed=True,
        physics={
            "enabled": True,
            "stabilization": {"iterations": 100},
            "barnesHut": {
                "gravitationalConstant": -8000,
                "centralGravity": 0.3,
                "springLength": 95,
                "springConstant": 0.04,
                "damping": 0.09
            }
        },
        interaction={
            "dragNodes": True,
            "dragView": True,
            "zoomView": True,
            "hover": True,
            "selectConnectedEdges": True
        },
        layout={
            "improvedLayout": True,
            "clusterThreshold": 150
        }
    )
    
    # Render the graph
    selected_node = agraph(nodes=nodes, edges=edges, config=config)
    
    # Show selected node details
    if selected_node:
        st.info(f"**Selected Node:** {selected_node}")
        
        # Find and display node details
        node_details = next((n for n in graph_data["nodes"] if n["id"] == selected_node), None)
        if node_details:
            st.json(node_details)


def render_hierarchical_graph(graph_data: Dict, show_labels: bool, color_scheme: str):
    """Render hierarchical graph using Plotly."""
    
    # Create NetworkX graph for layout calculation
    G = nx.DiGraph()
    
    for node in graph_data["nodes"]:
        G.add_node(node["id"], **node)
    
    for edge in graph_data["edges"]:
        G.add_edge(edge["source"], edge["target"], **edge)
    
    # Calculate hierarchical positions
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot')
    except:
        # Fallback to spring layout if graphviz not available
        pos = nx.spring_layout(G)
    
    # Create Plotly figure
    fig = go.Figure()
    
    # Add edges
    for edge in graph_data["edges"]:
        x0, y0 = pos[edge["source"]]
        x1, y1 = pos[edge["target"]]
        
        fig.add_trace(go.Scatter(
            x=[x0, x1, None],
            y=[y0, y1, None],
            mode='lines',
            line=dict(color='#999999', width=2),
            showlegend=False,
            hoverinfo='none'
        ))
    
    # Add nodes
    color_map = get_color_scheme(color_scheme)
    
    for node in graph_data["nodes"]:
        x, y = pos[node["id"]]
        node_color = color_map.get(node.get("type", "default"), "#1f77b4")
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=node.get("size", 20),
                color=node_color,
                line=dict(width=2, color='white')
            ),
            text=node["label"] if show_labels else "",
            textposition="middle center",
            name=node["label"],
            hovertemplate=f"<b>{node['label']}</b><br>Type: {node.get('type', 'unknown')}<br>Status: {node.get('status', 'unknown')}<extra></extra>"
        ))
    
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=20,l=5,r=5,t=40),
        annotations=[
            dict(
                text="Hierarchical Layout",
                showarrow=False,
                xref="paper", yref="paper",
                x=0.005, y=-0.002,
                xanchor='left', yanchor='bottom',
                font=dict(size=14, color="#666666")
            )
        ],
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_circular_graph(graph_data: Dict, show_labels: bool, color_scheme: str):
    """Render circular graph layout."""
    
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    n_nodes = len(nodes)
    
    # Calculate circular positions
    pos = {}
    for i, node in enumerate(nodes):
        angle = 2 * math.pi * i / n_nodes
        x = math.cos(angle) * 100
        y = math.sin(angle) * 100
        pos[node["id"]] = (x, y)
    
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        if edge["source"] in pos and edge["target"] in pos:
            x0, y0 = pos[edge["source"]]
            x1, y1 = pos[edge["target"]]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(color='#999999', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
    
    # Add nodes
    color_map = get_color_scheme(color_scheme)
    
    for node in nodes:
        x, y = pos[node["id"]]
        node_color = color_map.get(node.get("type", "default"), "#1f77b4")
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=node.get("size", 15),
                color=node_color,
                line=dict(width=2, color='white')
            ),
            text=node["label"] if show_labels else "",
            textposition="middle center",
            name=node["label"],
            hovertemplate=f"<b>{node['label']}</b><br>Type: {node.get('type', 'unknown')}<extra></extra>"
        ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, scaleanchor="x", scaleratio=1),
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=20, b=20, l=20, r=20)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_tree_graph(graph_data: Dict, show_labels: bool, color_scheme: str):
    """Render tree layout graph."""
    st.info("🌳 Tree layout visualization - Showing hierarchical relationships")
    render_hierarchical_graph(graph_data, show_labels, color_scheme)


def render_grid_graph(graph_data: Dict, show_labels: bool, color_scheme: str):
    """Render grid layout graph."""
    nodes = graph_data["nodes"]
    edges = graph_data["edges"]
    
    # Calculate grid dimensions
    n_nodes = len(nodes)
    grid_size = math.ceil(math.sqrt(n_nodes))
    
    # Calculate grid positions
    pos = {}
    for i, node in enumerate(nodes):
        row = i // grid_size
        col = i % grid_size
        pos[node["id"]] = (col * 50, -row * 50)
    
    fig = go.Figure()
    
    # Add edges
    for edge in edges:
        if edge["source"] in pos and edge["target"] in pos:
            x0, y0 = pos[edge["source"]]
            x1, y1 = pos[edge["target"]]
            
            fig.add_trace(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(color='#999999', width=1),
                showlegend=False,
                hoverinfo='none'
            ))
    
    # Add nodes
    color_map = get_color_scheme(color_scheme)
    
    for node in nodes:
        x, y = pos[node["id"]]
        node_color = color_map.get(node.get("type", "default"), "#1f77b4")
        
        fig.add_trace(go.Scatter(
            x=[x],
            y=[y],
            mode='markers+text' if show_labels else 'markers',
            marker=dict(
                size=node.get("size", 20),
                color=node_color,
                line=dict(width=2, color='white')
            ),
            text=node["label"] if show_labels else "",
            textposition="middle center",
            name=node["label"]
        ))
    
    fig.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=True, zeroline=False, showticklabels=False),
        plot_bgcolor='rgba(0,0,0,0)'
    )
    
    st.plotly_chart(fig, use_container_width=True)


def render_graph_analytics():
    """Render graph analytics dashboard."""
    st.subheader("📊 Graph Analytics & Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Connectivity Analysis")
        
        # Network metrics
        metrics_data = {
            "Metric": ["Density", "Clustering Coefficient", "Average Path Length", "Diameter", "Modularity"],
            "Value": [0.23, 0.67, 2.8, 5, 0.45],
            "Interpretation": [
                "Low density - sparse connections",
                "High clustering - community structure",
                "Short average paths - efficient",
                "Moderate diameter",
                "Good community separation"
            ]
        }
        
        import pandas as pd
        df_metrics = pd.DataFrame(metrics_data)
        st.dataframe(df_metrics, use_container_width=True)
        
        # Degree distribution
        degrees = [random.randint(1, 10) for _ in range(50)]
        fig_degree = px.histogram(x=degrees, nbins=10, title="Node Degree Distribution")
        fig_degree.update_layout(xaxis_title="Degree", yaxis_title="Count")
        st.plotly_chart(fig_degree, use_container_width=True)
    
    with col2:
        st.subheader("⚡ Performance Metrics")
        
        # Performance over time
        times = pd.date_range(start=pd.Timestamp.now() - pd.Timedelta(days=29), periods=30, freq='D')
        response_times = [random.uniform(100, 500) for _ in range(30)]  # Would come from real metrics
        throughput = [random.uniform(50, 150) for _ in range(30)]  # Would come from real metrics
        
        fig_perf = go.Figure()
        fig_perf.add_trace(go.Scatter(
            x=times, y=response_times, name="Response Time (ms)",
            line=dict(color='#ff7f0e')
        ))
        
        fig_perf2 = fig_perf
        fig_perf2.add_trace(go.Scatter(
            x=times, y=throughput, name="Throughput (req/s)",
            line=dict(color='#2ca02c'), yaxis="y2"
        ))
        
        fig_perf2.update_layout(
            title="Performance Metrics Over Time",
            xaxis_title="Date",
            yaxis=dict(title="Response Time (ms)", side="left"),
            yaxis2=dict(title="Throughput (req/s)", side="right", overlaying="y")
        )
        
        st.plotly_chart(fig_perf2, use_container_width=True)
        
        # Node importance ranking
        importance_data = {
            "Node": ["Security Agent", "IAM Service", "GCP Service", "Analytics", "Monitoring"],
            "Centrality": [0.95, 0.87, 0.82, 0.76, 0.71],
            "Influence": [0.92, 0.85, 0.79, 0.73, 0.68]
        }
        
        df_importance = pd.DataFrame(importance_data)
        fig_importance = px.bar(
            df_importance, x="Node", y=["Centrality", "Influence"],
            title="Node Importance Ranking",
            barmode="group"
        )
        st.plotly_chart(fig_importance, use_container_width=True)


def render_agent_coordination():
    """Render agent coordination analysis."""
    st.subheader("🎯 Multi-Agent Coordination Analysis")
    
    # Agent collaboration matrix
    agents = ["Security Agent", "Analytics Agent", "IAM Agent", "Monitor Agent", "Response Agent"]
    
    # Generate collaboration matrix
    import numpy as np
    collaboration_matrix = np.random.rand(len(agents), len(agents))
    np.fill_diagonal(collaboration_matrix, 1.0)  # Agents always collaborate with themselves
    
    fig_heatmap = px.imshow(
        collaboration_matrix,
        x=agents,
        y=agents,
        title="Agent Collaboration Intensity Matrix",
        color_continuous_scale="Viridis",
        aspect="auto"
    )
    
    fig_heatmap.update_layout(
        xaxis_title="Target Agent",
        yaxis_title="Source Agent"
    )
    
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔄 Agent Communication Flow")
        
        # Communication flow data
        flow_data = {
            "Source": ["Security Agent", "Security Agent", "Analytics Agent", "IAM Agent", "Monitor Agent"],
            "Target": ["Analytics Agent", "IAM Agent", "Monitor Agent", "Response Agent", "Security Agent"],
            "Messages": [156, 89, 234, 45, 167],
            "Bandwidth": [2.3, 1.8, 4.1, 0.9, 2.7]
        }
        
        df_flow = pd.DataFrame(flow_data)
        fig_flow = px.bar(df_flow, x="Source", y="Messages", 
                         title="Inter-Agent Message Volume",
                         color="Bandwidth", color_continuous_scale="Blues")
        st.plotly_chart(fig_flow, use_container_width=True)
        
    with col2:
        st.subheader("⏱️ Agent Response Times")
        
        # Response time distribution
        response_times = {
            agent: [random.normalvariate(200, 50) for _ in range(100)]
            for agent in agents
        }
        
        fig_response = go.Figure()
        for agent, times in response_times.items():
            fig_response.add_trace(go.Violin(
                y=times,
                name=agent,
                box_visible=True,
                meanline_visible=True
            ))
        
        fig_response.update_layout(
            title="Agent Response Time Distribution",
            yaxis_title="Response Time (ms)",
            showlegend=True
        )
        
        st.plotly_chart(fig_response, use_container_width=True)


def render_risk_analysis():
    """Render risk propagation analysis."""
    st.subheader("⚠️ Risk Propagation Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Risk Heat Map")
        
        # Risk assessment matrix
        services = ["IAM", "Storage", "Compute", "Network", "Security", "Monitoring"]
        risk_categories = ["Authentication", "Authorization", "Data", "Network", "Compliance"]
        
        risk_matrix = np.random.rand(len(risk_categories), len(services)) * 100
        
        fig_risk = px.imshow(
            risk_matrix,
            x=services,
            y=risk_categories,
            title="Security Risk Assessment Matrix",
            color_continuous_scale="Reds",
            aspect="auto"
        )
        
        fig_risk.update_traces(text=np.around(risk_matrix, 0), texttemplate="%{text}")
        st.plotly_chart(fig_risk, use_container_width=True)
        
    with col2:
        st.subheader("📈 Risk Propagation Paths")
        
        # Risk propagation simulation
        propagation_data = {
            "Step": list(range(1, 11)),
            "Affected_Services": [1, 2, 3, 5, 7, 10, 12, 15, 17, 18],
            "Risk_Score": [10, 25, 45, 67, 82, 90, 95, 98, 99, 100]
        }
        
        df_prop = pd.DataFrame(propagation_data)
        
        fig_prop = go.Figure()
        fig_prop.add_trace(go.Scatter(
            x=df_prop["Step"],
            y=df_prop["Affected_Services"],
            name="Affected Services",
            line=dict(color='#ff7f0e')
        ))
        
        fig_prop.add_trace(go.Scatter(
            x=df_prop["Step"],
            y=df_prop["Risk_Score"],
            name="Risk Score",
            line=dict(color='#d62728'),
            yaxis="y2"
        ))
        
        fig_prop.update_layout(
            title="Risk Propagation Over Time",
            xaxis_title="Propagation Step",
            yaxis=dict(title="Affected Services", side="left"),
            yaxis2=dict(title="Risk Score", side="right", overlaying="y")
        )
        
        st.plotly_chart(fig_prop, use_container_width=True)
    
    # Critical path analysis
    st.subheader("🛤️ Critical Risk Paths")
    
    critical_paths = [
        {"Path": "IAM → Storage → Compute", "Risk": 95, "Impact": "High", "Mitigation": "Enable MFA, encrypt data"},
        {"Path": "Network → Security → Monitoring", "Risk": 87, "Impact": "Medium", "Mitigation": "Firewall rules, alerts"},
        {"Path": "Storage → Network → Compute", "Risk": 76, "Impact": "High", "Mitigation": "VPC isolation, monitoring"}
    ]
    
    df_paths = pd.DataFrame(critical_paths)
    
    # Display as colored table
    def color_risk(val):
        if val >= 90:
            return 'background-color: #ffcccc'
        elif val >= 70:
            return 'background-color: #ffffcc'
        else:
            return 'background-color: #ccffcc'
    
    styled_df = df_paths.style.applymap(color_risk, subset=['Risk'])
    st.dataframe(styled_df, use_container_width=True)


def render_graph_explorer():
    """Render interactive graph explorer."""
    st.subheader("🔍 Interactive Graph Explorer")
    
    # Search and filter controls
    col1, col2, col3 = st.columns(3)
    
    with col1:
        search_query = st.text_input("🔎 Search Nodes", placeholder="Enter node name...")
    
    with col2:
        node_type_filter = st.multiselect(
            "Filter by Type",
            ["agent", "service", "api", "resource", "endpoint"],
            default=[]
        )
    
    with col3:
        status_filter = st.multiselect(
            "Filter by Status",
            ["running", "error", "disabled", "starting"],
            default=[]
        )
    
    # Path finding
    st.subheader("🗺️ Path Analysis")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        source_node = st.selectbox("Source Node", ["Security Agent", "IAM Service", "Analytics", "Monitoring"])
    
    with col2:
        target_node = st.selectbox("Target Node", ["GCP Service", "Storage API", "Compute API", "Network API"])
        
    with col3:
        if st.button("Find Shortest Path", type="primary"):
            st.success(f"Shortest path: {source_node} → IAM Service → GCP Service → {target_node}")
            st.info("Path length: 3 hops, Total latency: ~450ms")
    
    # Node details panel
    st.subheader("📋 Node Details")
    
    selected_node_detail = st.selectbox(
        "Select Node for Details",
        ["Security Agent", "IAM Service", "Analytics Service", "GCP Service", "Monitoring Service"]
    )
    
    # Dynamic node details - would be populated from backend
    node_details = {
        "Security Agent": {
            "Type": "AI Agent",
            "Status": "Running",
            "CPU Usage": "Monitoring...",
            "Memory": "Calculating...",
            "Connections": "Active",
            "Last Activity": "Real-time",
            "Version": "v2.1.0"
        },
        "IAM Service": {
            "Type": "Security Service",
            "Status": "Running", 
            "CPU Usage": "Monitoring...",
            "Memory": "Calculating...",
            "Connections": "Active",
            "Last Activity": "Real-time",
            "Version": "v1.8.3"
        }
    }
    
    if selected_node_detail in node_details:
        details = node_details[selected_node_detail]
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Status", details["Status"])
            st.metric("Type", details["Type"])
        
        with col2:
            st.metric("CPU Usage", details["CPU Usage"])
            st.metric("Memory", details["Memory"])
        
        with col3:
            st.metric("Connections", details["Connections"])
            st.metric("Version", details["Version"])
            
        with col4:
            st.metric("Last Activity", details["Last Activity"])
            
            if st.button(f"Restart {selected_node_detail}", key=f"restart_{selected_node_detail}"):
                st.success(f"Restart signal sent to {selected_node_detail}")


def generate_service_dependency_graph() -> Dict[str, List[Dict]]:
    """Generate service dependency graph data."""
    nodes = [
        {"id": "security_agent", "label": "Security Agent", "type": "agent", "size": 30, "status": "running"},
        {"id": "iam_service", "label": "IAM Service", "type": "service", "size": 25, "status": "running"},
        {"id": "gcp_service", "label": "GCP Service", "type": "service", "size": 25, "status": "running"},
        {"id": "analytics", "label": "Analytics Service", "type": "service", "size": 20, "status": "running"},
        {"id": "monitoring", "label": "Monitoring", "type": "service", "size": 20, "status": "running"},
        {"id": "logging", "label": "Cloud Logging", "type": "service", "size": 18, "status": "running"},
        {"id": "storage", "label": "Storage API", "type": "api", "size": 15, "status": "running"},
        {"id": "compute", "label": "Compute API", "type": "api", "size": 15, "status": "running"},
        {"id": "network", "label": "Network API", "type": "api", "size": 15, "status": "running"}
    ]
    
    edges = [
        {"source": "security_agent", "target": "iam_service", "weight": 3},
        {"source": "security_agent", "target": "gcp_service", "weight": 3},
        {"source": "security_agent", "target": "analytics", "weight": 2},
        {"source": "iam_service", "target": "gcp_service", "weight": 2},
        {"source": "gcp_service", "target": "storage", "weight": 2},
        {"source": "gcp_service", "target": "compute", "weight": 2},
        {"source": "gcp_service", "target": "network", "weight": 1},
        {"source": "analytics", "target": "monitoring", "weight": 2},
        {"source": "monitoring", "target": "logging", "weight": 2}
    ]
    
    return {"nodes": nodes, "edges": edges}


def generate_agent_collaboration_graph() -> Dict[str, List[Dict]]:
    """Generate agent collaboration graph data."""
    nodes = [
        {"id": "security_agent", "label": "Security Agent", "type": "agent", "size": 35, "speciality": "security"},
        {"id": "analytics_agent", "label": "Analytics Agent", "type": "agent", "size": 30, "speciality": "analysis"},
        {"id": "iam_agent", "label": "IAM Agent", "type": "agent", "size": 28, "speciality": "identity"},
        {"id": "monitor_agent", "label": "Monitor Agent", "type": "agent", "size": 25, "speciality": "monitoring"},
        {"id": "response_agent", "label": "Response Agent", "type": "agent", "size": 23, "speciality": "incident"},
        {"id": "orchestrator", "label": "Orchestrator", "type": "coordinator", "size": 40, "speciality": "coordination"}
    ]
    
    edges = [
        {"source": "orchestrator", "target": "security_agent", "weight": 3, "type": "coordination"},
        {"source": "orchestrator", "target": "analytics_agent", "weight": 3, "type": "coordination"},
        {"source": "orchestrator", "target": "iam_agent", "weight": 2, "type": "coordination"},
        {"source": "security_agent", "target": "analytics_agent", "weight": 2, "type": "collaboration"},
        {"source": "security_agent", "target": "iam_agent", "weight": 3, "type": "collaboration"},
        {"source": "analytics_agent", "target": "monitor_agent", "weight": 2, "type": "collaboration"},
        {"source": "security_agent", "target": "response_agent", "weight": 2, "type": "escalation"},
        {"source": "iam_agent", "target": "response_agent", "weight": 1, "type": "alert"}
    ]
    
    return {"nodes": nodes, "edges": edges}


def generate_risk_propagation_graph() -> Dict[str, List[Dict]]:
    """Generate risk propagation graph data."""
    nodes = [
        {"id": "iam_vuln", "label": "IAM Vulnerability", "type": "risk", "size": 35, "risk_level": "critical"},
        {"id": "storage_exposed", "label": "Storage Exposed", "type": "risk", "size": 30, "risk_level": "high"},
        {"id": "network_breach", "label": "Network Breach", "type": "risk", "size": 28, "risk_level": "high"},
        {"id": "data_leak", "label": "Data Leak", "type": "consequence", "size": 25, "risk_level": "critical"},
        {"id": "compliance_violation", "label": "Compliance Violation", "type": "consequence", "size": 23, "risk_level": "medium"},
        {"id": "service_disruption", "label": "Service Disruption", "type": "consequence", "size": 20, "risk_level": "medium"}
    ]
    
    edges = [
        {"source": "iam_vuln", "target": "storage_exposed", "weight": 3, "propagation_probability": 0.8},
        {"source": "iam_vuln", "target": "network_breach", "weight": 2, "propagation_probability": 0.6},
        {"source": "storage_exposed", "target": "data_leak", "weight": 3, "propagation_probability": 0.9},
        {"source": "network_breach", "target": "service_disruption", "weight": 2, "propagation_probability": 0.7},
        {"source": "data_leak", "target": "compliance_violation", "weight": 2, "propagation_probability": 0.85}
    ]
    
    return {"nodes": nodes, "edges": edges}


def generate_api_dependency_graph() -> Dict[str, List[Dict]]:
    """Generate API dependency graph data."""
    return generate_service_dependency_graph()  # Reuse for simplicity


def generate_multi_agent_workflow_graph() -> Dict[str, List[Dict]]:
    """Generate multi-agent workflow graph data."""
    nodes = [
        {"id": "user_request", "label": "User Request", "type": "input", "size": 20, "stage": "start"},
        {"id": "orchestrator", "label": "Orchestrator", "type": "coordinator", "size": 35, "stage": "coordination"},
        {"id": "security_scan", "label": "Security Scan", "type": "agent", "size": 25, "stage": "analysis"},
        {"id": "iam_check", "label": "IAM Check", "type": "agent", "size": 25, "stage": "analysis"},
        {"id": "risk_assess", "label": "Risk Assessment", "type": "agent", "size": 23, "stage": "evaluation"},
        {"id": "generate_report", "label": "Generate Report", "type": "agent", "size": 20, "stage": "output"},
        {"id": "user_response", "label": "User Response", "type": "output", "size": 18, "stage": "end"}
    ]
    
    edges = [
        {"source": "user_request", "target": "orchestrator", "weight": 1, "sequence": 1},
        {"source": "orchestrator", "target": "security_scan", "weight": 2, "sequence": 2},
        {"source": "orchestrator", "target": "iam_check", "weight": 2, "sequence": 2},
        {"source": "security_scan", "target": "risk_assess", "weight": 2, "sequence": 3},
        {"source": "iam_check", "target": "risk_assess", "weight": 2, "sequence": 3},
        {"source": "risk_assess", "target": "generate_report", "weight": 2, "sequence": 4},
        {"source": "generate_report", "target": "user_response", "weight": 1, "sequence": 5}
    ]
    
    return {"nodes": nodes, "edges": edges}


def get_color_scheme(scheme: str) -> Dict[str, str]:
    """Get color mapping for different schemes."""
    schemes = {
        "Default": {
            "agent": "#1f77b4",
            "service": "#ff7f0e", 
            "api": "#2ca02c",
            "resource": "#d62728",
            "coordinator": "#9467bd",
            "risk": "#8c564b",
            "consequence": "#e377c2",
            "input": "#7f7f7f",
            "output": "#bcbd22",
            "edge": "#999999"
        },
        "Security Risk": {
            "agent": "#28a745",
            "service": "#ffc107",
            "api": "#17a2b8",
            "resource": "#6c757d",
            "coordinator": "#007bff",
            "risk": "#dc3545",
            "consequence": "#fd7e14",
            "input": "#6c757d",
            "output": "#28a745",
            "edge": "#6c757d"
        },
        "Performance": {
            "agent": "#00ff00",
            "service": "#ffff00",
            "api": "#ff9900",
            "resource": "#ff0000",
            "coordinator": "#0099ff",
            "risk": "#ff0000",
            "consequence": "#ff6600",
            "input": "#cccccc",
            "output": "#00ff00",
            "edge": "#666666"
        },
        "Service Status": {
            "running": "#28a745",
            "error": "#dc3545", 
            "disabled": "#6c757d",
            "starting": "#ffc107",
            "agent": "#007bff",
            "service": "#28a745",
            "api": "#17a2b8",
            "edge": "#adb5bd"
        },
        "Dark Theme": {
            "agent": "#64b5f6",
            "service": "#81c784",
            "api": "#ffb74d",
            "resource": "#f06292",
            "coordinator": "#ba68c8",
            "risk": "#e57373",
            "consequence": "#ff8a65",
            "input": "#90a4ae",
            "output": "#aed581",
            "edge": "#616161"
        }
    }
    
    return schemes.get(scheme, schemes["Default"])


def calculate_connected_components(graph_data: Dict) -> int:
    """Calculate number of connected components in graph."""
    # Simple approximation
    return max(1, len(graph_data["nodes"]) // 5)


def calculate_max_degree(graph_data: Dict) -> int:
    """Calculate maximum degree in graph."""
    degree_count = {}
    
    for edge in graph_data["edges"]:
        degree_count[edge["source"]] = degree_count.get(edge["source"], 0) + 1
        degree_count[edge["target"]] = degree_count.get(edge["target"], 0) + 1
    
    return max(degree_count.values()) if degree_count else 0