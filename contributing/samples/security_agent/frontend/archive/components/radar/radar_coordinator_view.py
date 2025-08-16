"""
RADAR Coordinator View - Main orchestration interface for RADAR workflow.

This module provides the central coordination view for the RADAR methodology,
displaying the workflow visually and managing phase transitions.
"""

import streamlit as st
import logging
from typing import Dict, Any, Optional, List
from datetime import datetime
import time

from .radar_state_manager import radar_state_manager, RADARPhase, RADARContext

logger = logging.getLogger(__name__)


def render_radar_coordinator_view():
    """
    Main entry point for RADAR coordinator view.
    
    This view provides:
    - Visual workflow representation
    - Phase status tracking
    - Quick access to individual phases
    - Real-time progress monitoring
    """
    st.title("🎯 RADAR Security Analysis Coordinator")
    
    # Initialize or get context
    context = radar_state_manager.get_context()
    if not context:
        project_id = st.session_state.get('selected_project', 'default-project')
        user_id = st.session_state.get('current_user', {}).get('email', 'default')
        context = radar_state_manager.initialize_context(project_id, user_id)
    
    # Display workflow header
    render_workflow_header(context)
    
    # Main layout columns
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Visual workflow diagram
        render_workflow_diagram(context)
        
        # Phase control panel
        render_phase_controls(context)
    
    with col2:
        # Context panel
        render_context_panel(context)
        
        # Quick actions
        render_quick_actions(context)
    
    # Progress timeline
    render_progress_timeline(context)
    
    # Export/Import controls
    render_export_import_controls()


def render_workflow_header(context: RADARContext):
    """Render workflow header with status information."""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Project", context.project_id)
    
    with col2:
        completed = len(radar_state_manager.get_completed_phases())
        st.metric("Progress", f"{completed}/5 Phases")
    
    with col3:
        current_phase = context.current_phase
        if current_phase:
            st.metric("Current Phase", current_phase.value.title())
        else:
            st.metric("Current Phase", "Not Started")
    
    with col4:
        elapsed = (datetime.now() - context.created_at).seconds // 60
        st.metric("Elapsed Time", f"{elapsed} min")


def render_workflow_diagram(context: RADARContext):
    """Render visual RADAR workflow diagram."""
    st.subheader("📊 RADAR Workflow")
    
    # Create workflow visualization using columns
    phases = list(RADARPhase)
    cols = st.columns(5)
    
    phase_icons = {
        RADARPhase.RECOGNITION: "🔍",
        RADARPhase.ASSESSMENT: "🛡️",
        RADARPhase.DECISION: "🎯",
        RADARPhase.ACTION: "⚡",
        RADARPhase.REVIEW: "📊"
    }
    
    phase_descriptions = {
        RADARPhase.RECOGNITION: "Discover Resources",
        RADARPhase.ASSESSMENT: "Evaluate Security",
        RADARPhase.DECISION: "Prioritize Issues",
        RADARPhase.ACTION: "Execute Fixes",
        RADARPhase.REVIEW: "Verify & Report"
    }
    
    for idx, (col, phase) in enumerate(zip(cols, phases)):
        with col:
            phase_result = context.phases.get(phase)
            
            # Determine status color
            if phase_result:
                if phase_result.status == "completed":
                    status_color = "✅"
                    button_type = "primary"
                elif phase_result.status == "in_progress":
                    status_color = "🔄"
                    button_type = "secondary"
                elif phase_result.status == "failed":
                    status_color = "❌"
                    button_type = "secondary"
                else:
                    status_color = "⏳"
                    button_type = "secondary"
            else:
                status_color = "⏳"
                button_type = "secondary"
            
            # Phase card
            with st.container():
                st.markdown(f"### {phase_icons[phase]}")
                st.markdown(f"**{phase.value.title()}**")
                st.caption(phase_descriptions[phase])
                st.markdown(f"Status: {status_color}")
                
                # Navigate button
                if st.button(
                    "Open Phase",
                    key=f"nav_{phase.value}",
                    type=button_type,
                    use_container_width=True,
                    disabled=not radar_state_manager.can_execute_phase(phase)
                ):
                    st.session_state.radar_current_phase = phase
                    st.session_state.page = f"radar_{phase.value}"
                    st.rerun()
            
            # Draw arrow (except for last phase)
            if idx < len(phases) - 1:
                st.markdown("→")


def render_phase_controls(context: RADARContext):
    """Render phase execution controls."""
    st.subheader("🎮 Phase Controls")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Start sequential workflow
        if st.button("▶️ Start Full Analysis", use_container_width=True):
            start_sequential_workflow(context)
    
    with col2:
        # Reset workflow
        if st.button("🔄 Reset Workflow", use_container_width=True):
            radar_state_manager.reset_context()
            st.rerun()
    
    with col3:
        # Auto-advance toggle
        auto_advance = st.checkbox("Auto-advance phases", value=False)
        st.session_state.radar_auto_advance = auto_advance


def render_context_panel(context: RADARContext):
    """Render context information panel."""
    st.subheader("📋 Context Information")
    
    # Session info
    with st.expander("Session Details", expanded=False):
        st.json({
            "session_id": context.session_id,
            "project_id": context.project_id,
            "user_id": context.user_id,
            "created_at": context.created_at.isoformat(),
            "updated_at": context.updated_at.isoformat()
        })
    
    # Completed phases summary
    completed_phases = radar_state_manager.get_completed_phases()
    if completed_phases:
        st.markdown("### ✅ Completed Phases")
        for phase in completed_phases:
            phase_result = context.phases[phase]
            if phase_result.results:
                with st.expander(f"{phase.value.title()}", expanded=False):
                    # Show summary of results
                    if isinstance(phase_result.results, dict):
                        for key, value in phase_result.results.items():
                            if isinstance(value, (str, int, float, bool)):
                                st.write(f"**{key}:** {value}")
                            elif isinstance(value, list) and len(value) > 0:
                                st.write(f"**{key}:** {len(value)} items")
                            elif isinstance(value, dict):
                                st.write(f"**{key}:** {len(value)} entries")
    
    # Pending phases
    pending_phases = radar_state_manager.get_pending_phases()
    if pending_phases:
        st.markdown("### ⏳ Pending Phases")
        for phase in pending_phases:
            deps = radar_state_manager.get_phase_dependencies(phase)
            if deps:
                st.write(f"- **{phase.value.title()}** (requires: {', '.join([d.value for d in deps])})")
            else:
                st.write(f"- **{phase.value.title()}** (ready to start)")


def render_quick_actions(context: RADARContext):
    """Render quick action buttons."""
    st.subheader("⚡ Quick Actions")
    
    # Determine available actions based on context
    completed_phases = radar_state_manager.get_completed_phases()
    
    if RADARPhase.RECOGNITION not in completed_phases:
        if st.button("🔍 Start Resource Discovery", use_container_width=True):
            st.session_state.radar_current_phase = RADARPhase.RECOGNITION
            st.session_state.page = "radar_recognition"
            st.rerun()
    
    elif RADARPhase.ASSESSMENT not in completed_phases:
        if st.button("🛡️ Assess Security Posture", use_container_width=True):
            st.session_state.radar_current_phase = RADARPhase.ASSESSMENT
            st.session_state.page = "radar_assessment"
            st.rerun()
    
    elif RADARPhase.DECISION not in completed_phases:
        if st.button("🎯 Generate Recommendations", use_container_width=True):
            st.session_state.radar_current_phase = RADARPhase.DECISION
            st.session_state.page = "radar_decision"
            st.rerun()
    
    elif RADARPhase.ACTION not in completed_phases:
        if st.button("⚡ Execute Remediation", use_container_width=True):
            st.session_state.radar_current_phase = RADARPhase.ACTION
            st.session_state.page = "radar_action"
            st.rerun()
    
    elif RADARPhase.REVIEW not in completed_phases:
        if st.button("📊 Generate Report", use_container_width=True):
            st.session_state.radar_current_phase = RADARPhase.REVIEW
            st.session_state.page = "radar_review"
            st.rerun()
    
    else:
        st.success("✅ All phases completed!")
        if st.button("📄 View Full Report", use_container_width=True):
            st.session_state.page = "radar_report"
            st.rerun()
    
    # Common actions
    st.markdown("---")
    
    if st.button("💬 Chat with Security Assistant", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()
    
    if st.button("📊 View Dashboard", use_container_width=True):
        st.session_state.page = "dashboard"
        st.rerun()


def render_progress_timeline(context: RADARContext):
    """Render progress timeline showing phase execution history."""
    st.subheader("📈 Execution Timeline")
    
    # Create timeline visualization
    timeline_data = []
    for phase in RADARPhase:
        phase_result = context.phases.get(phase)
        if phase_result and phase_result.start_time:
            timeline_data.append({
                "Phase": phase.value.title(),
                "Status": phase_result.status,
                "Start": phase_result.start_time.strftime("%H:%M:%S"),
                "End": phase_result.end_time.strftime("%H:%M:%S") if phase_result.end_time else "In Progress",
                "Duration": str(phase_result.end_time - phase_result.start_time).split('.')[0] if phase_result.end_time else "N/A"
            })
    
    if timeline_data:
        st.dataframe(timeline_data, use_container_width=True)
    else:
        st.info("No phases have been executed yet. Start the analysis to see the timeline.")


def render_export_import_controls():
    """Render export/import controls for context management."""
    st.subheader("💾 Context Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📥 Export Context", use_container_width=True):
            context_json = radar_state_manager.export_context()
            if context_json:
                st.download_button(
                    label="Download Context JSON",
                    data=context_json,
                    file_name=f"radar_context_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
            else:
                st.warning("No context to export")
    
    with col2:
        uploaded_file = st.file_uploader("Import Context", type="json")
        if uploaded_file:
            context_str = uploaded_file.read().decode("utf-8")
            if radar_state_manager.import_context(context_str):
                st.success("Context imported successfully!")
                st.rerun()
            else:
                st.error("Failed to import context")


def start_sequential_workflow(context: RADARContext):
    """Start sequential execution of all RADAR phases."""
    with st.spinner("Starting RADAR analysis workflow..."):
        # This would typically trigger backend execution
        # For now, we'll simulate starting the first phase
        if radar_state_manager.start_phase(RADARPhase.RECOGNITION):
            st.success("Started Recognition phase")
            time.sleep(1)
            st.session_state.radar_current_phase = RADARPhase.RECOGNITION
            st.session_state.page = "radar_recognition"
            st.rerun()
        else:
            st.error("Failed to start workflow. Check phase dependencies.")


def get_phase_color(status: str) -> str:
    """Get color for phase status."""
    colors = {
        "completed": "green",
        "in_progress": "blue",
        "failed": "red",
        "pending": "gray"
    }
    return colors.get(status, "gray")


def get_phase_icon(phase: RADARPhase) -> str:
    """Get icon for RADAR phase."""
    icons = {
        RADARPhase.RECOGNITION: "🔍",
        RADARPhase.ASSESSMENT: "🛡️",
        RADARPhase.DECISION: "🎯",
        RADARPhase.ACTION: "⚡",
        RADARPhase.REVIEW: "📊"
    }
    return icons.get(phase, "📋")