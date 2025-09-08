"""
Chart Components
===============

Reusable chart components using Plotly for consistent visualization.
"""

import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import pandas as pd
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta

class SecurityCharts:
    """Collection of security-specific chart components."""
    
    @staticmethod
    def render_security_score_gauge(score: float, title: str = "Security Score") -> go.Figure:
        """Render a gauge chart for security scores."""
        fig = go.Figure(go.Indicator(
            mode = "gauge+number+delta",
            value = score,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': title},
            delta = {'reference': 80},
            gauge = {
                'axis': {'range': [None, 100]},
                'bar': {'color': "darkblue"},
                'steps': [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 80], 'color': "gray"}
                ],
                'threshold': {
                    'line': {'color': "red", 'width': 4},
                    'thickness': 0.75,
                    'value': 90
                }
            }
        ))
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    @staticmethod
    def render_severity_distribution(data: List[Dict]) -> go.Figure:
        """Render severity distribution pie chart."""
        df = pd.DataFrame(data)
        
        colors = {
            'Critical': '#dc3545',
            'High': '#fd7e14',
            'Medium': '#ffc107',
            'Low': '#28a745',
            'Info': '#17a2b8'
        }
        
        fig = px.pie(
            df, 
            values='count', 
            names='severity',
            title='Security Findings by Severity',
            color='severity',
            color_discrete_map=colors
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    @staticmethod
    def render_timeline_chart(data: List[Dict], 
                            x_col: str = 'date', 
                            y_col: str = 'count',
                            title: str = "Security Findings Over Time") -> go.Figure:
        """Render timeline chart for security metrics."""
        df = pd.DataFrame(data)
        
        fig = px.line(
            df, 
            x=x_col, 
            y=y_col,
            title=title,
            markers=True
        )
        
        fig.update_traces(
            line=dict(width=3),
            marker=dict(size=8)
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis_title="Date",
            yaxis_title="Count"
        )
        
        return fig
    
    @staticmethod
    def render_resource_compliance_chart(data: List[Dict]) -> go.Figure:
        """Render compliance status chart."""
        df = pd.DataFrame(data)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            name='Compliant',
            x=df['resource_type'],
            y=df['compliant'],
            marker_color='green'
        ))
        
        fig.add_trace(go.Bar(
            name='Non-Compliant',
            x=df['resource_type'],
            y=df['non_compliant'],
            marker_color='red'
        ))
        
        fig.update_layout(
            barmode='stack',
            title='Compliance Status by Resource Type',
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis_title="Resource Type",
            yaxis_title="Count"
        )
        
        return fig
    
    @staticmethod
    def render_network_topology(data: List[Dict]) -> go.Figure:
        """Render network topology visualization."""
        # This is a simplified network graph
        # In production, you'd use networkx or similar for complex topologies
        
        fig = go.Figure()
        
        # Add nodes
        for node in data:
            fig.add_trace(go.Scatter(
                x=[node['x']],
                y=[node['y']],
                mode='markers+text',
                text=[node['name']],
                textposition='middle center',
                marker=dict(
                    size=30,
                    color=node.get('color', 'blue')
                ),
                showlegend=False
            ))
        
        # Add edges (connections)
        for edge in data:
            if 'connections' in edge:
                for connection in edge['connections']:
                    target = next(n for n in data if n['name'] == connection)
                    fig.add_trace(go.Scatter(
                        x=[edge['x'], target['x']],
                        y=[edge['y'], target['y']],
                        mode='lines',
                        line=dict(width=2, color='gray'),
                        showlegend=False
                    ))
        
        fig.update_layout(
            title='Network Topology',
            height=500,
            showlegend=False,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig

class MetricCharts:
    """General metric visualization components."""
    
    @staticmethod
    def render_kpi_cards(metrics: List[Dict]):
        """Render KPI metric cards in columns."""
        num_metrics = len(metrics)
        cols = st.columns(num_metrics)
        
        for i, metric in enumerate(metrics):
            with cols[i]:
                st.metric(
                    label=metric['label'],
                    value=metric['value'],
                    delta=metric.get('delta'),
                    delta_color=metric.get('delta_color', 'normal'),
                    help=metric.get('help')
                )
    
    @staticmethod
    def render_heatmap(data: pd.DataFrame, 
                      title: str = "Security Heatmap",
                      x_col: str = 'x',
                      y_col: str = 'y',
                      z_col: str = 'value') -> go.Figure:
        """Render heatmap visualization."""
        
        fig = px.imshow(
            data.pivot(index=y_col, columns=x_col, values=z_col),
            title=title,
            aspect='auto',
            color_continuous_scale='Reds'
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50)
        )
        
        return fig
    
    @staticmethod
    def render_multi_series_timeline(data: List[Dict], 
                                   series_col: str = 'series',
                                   x_col: str = 'date',
                                   y_col: str = 'value',
                                   title: str = "Multi-Series Timeline") -> go.Figure:
        """Render multi-series timeline chart."""
        df = pd.DataFrame(data)
        
        fig = px.line(
            df,
            x=x_col,
            y=y_col,
            color=series_col,
            title=title,
            markers=True
        )
        
        fig.update_layout(
            height=400,
            margin=dict(t=50, b=50, l=50, r=50),
            xaxis_title="Date",
            yaxis_title="Value"
        )
        
        return fig

class InteractiveCharts:
    """Interactive chart components with user controls."""
    
    @staticmethod
    def render_filterable_chart(data: pd.DataFrame, 
                              chart_type: str = 'line',
                              filter_columns: List[str] = None) -> go.Figure:
        """Render chart with interactive filters."""
        
        if filter_columns:
            # Add filter controls
            filters = {}
            cols = st.columns(len(filter_columns))
            
            for i, col in enumerate(filter_columns):
                with cols[i]:
                    unique_values = data[col].unique()
                    selected = st.multiselect(
                        f"Filter by {col}",
                        options=unique_values,
                        default=unique_values,
                        key=f"filter_{col}"
                    )
                    filters[col] = selected
            
            # Apply filters
            filtered_data = data.copy()
            for col, values in filters.items():
                if values:
                    filtered_data = filtered_data[filtered_data[col].isin(values)]
        else:
            filtered_data = data
        
        # Render chart based on type
        if chart_type == 'line':
            fig = px.line(filtered_data, x=filtered_data.columns[0], y=filtered_data.columns[1])
        elif chart_type == 'bar':
            fig = px.bar(filtered_data, x=filtered_data.columns[0], y=filtered_data.columns[1])
        elif chart_type == 'scatter':
            fig = px.scatter(filtered_data, x=filtered_data.columns[0], y=filtered_data.columns[1])
        else:
            fig = px.line(filtered_data, x=filtered_data.columns[0], y=filtered_data.columns[1])
        
        return fig