"""
Mobile-Optimized Layout Components
=================================

Specialized layout components optimized specifically for mobile devices
with touch-friendly interfaces, swipe navigation, and progressive disclosure.
"""

import streamlit as st
import streamlit.components.v1 as components
from typing import Dict, List, Optional, Any, Callable
from .base_layout import ResponsiveLayout
import json

class MobileLayout(ResponsiveLayout):
    """Mobile-optimized layout with touch-friendly components and gestures"""
    
    def __init__(self):
        super().__init__()
        self._setup_mobile_css()
    
    def _setup_mobile_css(self) -> None:
        """Setup mobile-specific CSS styles"""
        mobile_css = """
        <style>
        /* Mobile-first responsive design */
        @media (max-width: 768px) {
            /* Touch-friendly buttons */
            .stButton > button {
                min-height: 44px;
                font-size: 16px;
                padding: 12px 20px;
                border-radius: 8px;
                width: 100%;
            }
            
            /* Larger touch targets for inputs */
            .stSelectbox > div > div {
                min-height: 44px;
                font-size: 16px;
            }
            
            .stTextInput > div > div > input {
                min-height: 44px;
                font-size: 16px;
                padding: 12px;
            }
            
            /* Mobile-optimized metrics */
            .metric-container {
                background: #f8f9fa;
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            }
            
            /* Swipeable cards */
            .swipe-card {
                background: white;
                border-radius: 12px;
                padding: 20px;
                margin: 10px 0;
                box-shadow: 0 4px 12px rgba(0,0,0,0.1);
                transition: transform 0.3s ease;
                touch-action: pan-x;
            }
            
            .swipe-card:active {
                transform: scale(0.98);
            }
            
            /* Mobile navigation */
            .mobile-nav {
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                background: white;
                border-top: 1px solid #e0e0e0;
                padding: 10px;
                z-index: 1000;
                display: flex;
                justify-content: space-around;
            }
            
            .mobile-nav-item {
                flex: 1;
                text-align: center;
                padding: 8px;
                border-radius: 8px;
                cursor: pointer;
                transition: background-color 0.3s ease;
            }
            
            .mobile-nav-item:hover {
                background-color: #f5f5f5;
            }
            
            .mobile-nav-item.active {
                background-color: #e3f2fd;
                color: #1976d2;
            }
            
            /* Collapsible sections */
            .collapsible-section {
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                margin: 10px 0;
                overflow: hidden;
            }
            
            .collapsible-header {
                background: #f8f9fa;
                padding: 15px;
                cursor: pointer;
                display: flex;
                justify-content: space-between;
                align-items: center;
                font-weight: bold;
            }
            
            .collapsible-content {
                padding: 15px;
                display: none;
            }
            
            .collapsible-content.active {
                display: block;
            }
            
            /* Mobile-optimized tables */
            .mobile-table {
                display: block;
                width: 100%;
                overflow-x: auto;
                white-space: nowrap;
            }
            
            .mobile-table table {
                border-collapse: separate;
                border-spacing: 0;
                width: 100%;
            }
            
            .mobile-table th,
            .mobile-table td {
                padding: 12px 8px;
                text-align: left;
                border-bottom: 1px solid #e0e0e0;
                font-size: 14px;
            }
            
            /* Progress indicators */
            .progress-indicator {
                display: flex;
                justify-content: center;
                align-items: center;
                padding: 20px;
            }
            
            .progress-dot {
                width: 12px;
                height: 12px;
                border-radius: 50%;
                background-color: #e0e0e0;
                margin: 0 4px;
                transition: background-color 0.3s ease;
            }
            
            .progress-dot.active {
                background-color: #1976d2;
            }
            
            /* Floating action button */
            .fab {
                position: fixed;
                bottom: 80px;
                right: 20px;
                width: 56px;
                height: 56px;
                border-radius: 50%;
                background-color: #1976d2;
                color: white;
                border: none;
                font-size: 24px;
                cursor: pointer;
                box-shadow: 0 4px 12px rgba(25,118,210,0.3);
                z-index: 999;
                transition: transform 0.3s ease;
            }
            
            .fab:active {
                transform: scale(0.9);
            }
        }
        
        /* Haptic feedback simulation */
        .haptic-feedback {
            animation: haptic-pulse 0.1s ease-in-out;
        }
        
        @keyframes haptic-pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        </style>
        """
        
        st.markdown(mobile_css, unsafe_allow_html=True)
    
    def create_mobile_navigation(self, nav_items: List[Dict[str, Any]]) -> str:
        """Create mobile bottom navigation"""
        nav_html = """
        <div class="mobile-nav">
        """
        
        for item in nav_items:
            active_class = "active" if item.get('active', False) else ""
            nav_html += f"""
            <div class="mobile-nav-item {active_class}" onclick="selectNavItem('{item['key']}')">
                <div style="font-size: 20px;">{item.get('icon', '📱')}</div>
                <div style="font-size: 12px; margin-top: 4px;">{item['label']}</div>
            </div>
            """
        
        nav_html += """
        </div>
        <script>
        function selectNavItem(key) {
            // Remove active class from all items
            document.querySelectorAll('.mobile-nav-item').forEach(item => {
                item.classList.remove('active');
            });
            
            // Add active class to clicked item
            event.target.closest('.mobile-nav-item').classList.add('active');
            
            // Send selection back to Streamlit
            window.parent.postMessage({type: 'nav_selection', key: key}, '*');
        }
        </script>
        """
        
        components.html(nav_html, height=80)
        
        # Return selected navigation item
        return st.session_state.get('mobile_nav_selection', nav_items[0]['key'] if nav_items else None)
    
    def create_swipeable_cards(self, cards_data: List[Dict[str, Any]]) -> None:
        """Create swipeable cards for mobile"""
        cards_html = """
        <div style="margin-bottom: 20px;">
        """
        
        for i, card in enumerate(cards_data):
            cards_html += f"""
            <div class="swipe-card" id="card-{i}">
                <h3 style="margin: 0 0 10px 0; color: #333;">{card.get('title', 'Card')}</h3>
                <p style="margin: 0 0 15px 0; color: #666;">{card.get('description', '')}</p>
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <span style="font-weight: bold; color: #1976d2;">{card.get('value', '')}</span>
                    <span style="font-size: 12px; color: #999;">{card.get('timestamp', '')}</span>
                </div>
            </div>
            """
        
        cards_html += """
        </div>
        <script>
        // Add swipe gesture support
        document.querySelectorAll('.swipe-card').forEach(card => {
            let startX = 0;
            let currentX = 0;
            let isDragging = false;
            
            card.addEventListener('touchstart', (e) => {
                startX = e.touches[0].clientX;
                isDragging = true;
            });
            
            card.addEventListener('touchmove', (e) => {
                if (!isDragging) return;
                currentX = e.touches[0].clientX;
                const diffX = currentX - startX;
                card.style.transform = `translateX(${diffX}px)`;
            });
            
            card.addEventListener('touchend', () => {
                if (!isDragging) return;
                isDragging = false;
                
                const diffX = currentX - startX;
                if (Math.abs(diffX) > 100) {
                    // Swipe detected
                    card.style.transform = `translateX(${diffX > 0 ? '100%' : '-100%'})`;
                    setTimeout(() => {
                        card.style.display = 'none';
                    }, 300);
                } else {
                    // Reset position
                    card.style.transform = 'translateX(0)';
                }
            });
        });
        </script>
        """
        
        components.html(cards_html, height=len(cards_data) * 120 + 50)
    
    def create_mobile_metrics_grid(self, metrics: List[Dict[str, Any]]) -> None:
        """Create mobile-optimized metrics grid"""
        for metric in metrics:
            metric_html = f"""
            <div class="metric-container">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <div style="font-size: 12px; color: #666; text-transform: uppercase; margin-bottom: 4px;">
                            {metric.get('label', 'Metric')}
                        </div>
                        <div style="font-size: 24px; font-weight: bold; color: #333;">
                            {metric.get('value', '0')}
                        </div>
                        {f'<div style="font-size: 12px; color: {"green" if str(metric.get("delta", "")).startswith("+") else "red"};">{metric.get("delta", "")}</div>' if metric.get('delta') else ''}
                    </div>
                    <div style="font-size: 32px; opacity: 0.6;">
                        {metric.get('icon', '📊')}
                    </div>
                </div>
            </div>
            """
            
            st.markdown(metric_html, unsafe_allow_html=True)
    
    def create_progressive_disclosure(self, sections: List[Dict[str, Any]]) -> None:
        """Create progressive disclosure interface for mobile"""
        disclosure_html = """
        <div>
        """
        
        for i, section in enumerate(sections):
            disclosure_html += f"""
            <div class="collapsible-section">
                <div class="collapsible-header" onclick="toggleSection({i})">
                    <span>{section.get('title', 'Section')}</span>
                    <span id="arrow-{i}" style="transition: transform 0.3s ease;">▼</span>
                </div>
                <div class="collapsible-content" id="content-{i}">
            """
            
            # Add section content
            if section.get('type') == 'metrics':
                for metric in section.get('data', []):
                    disclosure_html += f"""
                    <div style="padding: 8px 0; border-bottom: 1px solid #f0f0f0;">
                        <div style="font-weight: bold;">{metric.get('label', '')}</div>
                        <div>{metric.get('value', '')}</div>
                    </div>
                    """
            elif section.get('type') == 'list':
                for item in section.get('data', []):
                    disclosure_html += f"""
                    <div style="padding: 8px 0; border-bottom: 1px solid #f0f0f0;">
                        • {item}
                    </div>
                    """
            else:
                disclosure_html += f"<div>{section.get('content', 'No content')}</div>"
            
            disclosure_html += """
                </div>
            </div>
            """
        
        disclosure_html += """
        </div>
        <script>
        function toggleSection(index) {
            const content = document.getElementById(`content-${index}`);
            const arrow = document.getElementById(`arrow-${index}`);
            
            if (content.classList.contains('active')) {
                content.classList.remove('active');
                arrow.style.transform = 'rotate(0deg)';
            } else {
                content.classList.add('active');
                arrow.style.transform = 'rotate(180deg)';
            }
        }
        </script>
        """
        
        components.html(disclosure_html, height=len(sections) * 60 + 100)
    
    def create_touch_friendly_chart(self, chart_data: Dict[str, Any], chart_type: str = 'line') -> None:
        """Create touch-friendly charts optimized for mobile interaction"""
        # Mobile-specific chart configuration
        mobile_config = {
            'displayModeBar': False,  # Hide toolbar on mobile
            'responsive': True,
            'doubleClick': 'reset',
            'showTips': True,
            'scrollZoom': False,  # Disable scroll zoom to prevent page scroll conflicts
            'dragMode': False,     # Disable drag to prevent gesture conflicts
        }
        
        try:
            import plotly.express as px
            import plotly.graph_objects as go
            
            if chart_type == 'line':
                fig = px.line(
                    chart_data.get('data', []),
                    x=chart_data.get('x', 'x'),
                    y=chart_data.get('y', 'y'),
                    title=chart_data.get('title', 'Chart'),
                    height=250  # Fixed mobile height
                )
            elif chart_type == 'bar':
                fig = px.bar(
                    chart_data.get('data', []),
                    x=chart_data.get('x', 'x'),
                    y=chart_data.get('y', 'y'),
                    title=chart_data.get('title', 'Chart'),
                    height=250
                )
            else:
                fig = px.scatter(
                    chart_data.get('data', []),
                    x=chart_data.get('x', 'x'),
                    y=chart_data.get('y', 'y'),
                    title=chart_data.get('title', 'Chart'),
                    height=250
                )
            
            # Mobile-optimized layout
            fig.update_layout(
                font=dict(size=10),
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False,
                title_font_size=14,
                xaxis_title_font_size=10,
                yaxis_title_font_size=10,
            )
            
            st.plotly_chart(fig, use_container_width=True, config=mobile_config, key="mobile_touch_friendly_chart")
            
        except Exception as e:
            st.error(f"Error creating mobile chart: {str(e)}")
    
    def create_floating_action_button(self, action_config: Dict[str, Any]) -> bool:
        """Create floating action button for primary mobile actions"""
        fab_html = f"""
        <button class="fab" onclick="triggerFabAction()" title="{action_config.get('tooltip', 'Action')}">
            {action_config.get('icon', '+')}
        </button>
        <script>
        function triggerFabAction() {
            // Add haptic feedback simulation
            document.querySelector('.fab').classList.add('haptic-feedback');
            setTimeout(() => {
                document.querySelector('.fab').classList.remove('haptic-feedback');
            }, 100);
            
            // Send action back to Streamlit
            window.parent.postMessage({{
                type: 'fab_action', 
                action: '{action_config.get('action', 'default')}'
            }}, '*');
        }
        </script>
        """
        
        components.html(fab_html, height=0)
        
        # Check if FAB was clicked
        return st.session_state.get('fab_clicked', False)
    
    def create_mobile_data_table(self, df, title: str = None) -> None:
        """Create mobile-optimized data table with horizontal scrolling"""
        if title:
            st.subheader(title)
        
        if df.empty:
            st.info("No data available")
            return
        
        # Convert dataframe to HTML with mobile styling
        table_html = f"""
        <div class="mobile-table">
            {df.to_html(classes='mobile-table-content', table_id='mobile-data-table', escape=False)}
        </div>
        <script>
        // Add touch scroll indicators
        const table = document.getElementById('mobile-data-table');
        if (table && table.scrollWidth > table.clientWidth) {
            const indicator = document.createElement('div');
            indicator.innerHTML = '← Swipe to see more →';
            indicator.style.textAlign = 'center';
            indicator.style.fontSize = '12px';
            indicator.style.color = '#666';
            indicator.style.padding = '10px';
            table.parentNode.appendChild(indicator);
        }
        </script>
        """
        
        components.html(table_html, height=min(len(df) * 40 + 100, 400))

# Global mobile layout instance
mobile_layout = MobileLayout()
