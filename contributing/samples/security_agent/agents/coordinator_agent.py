"""
ADK Coordinator Agent with LLM-Driven Delegation
Implements proper agent transfer pattern following ADK multi-agent architecture
"""

from google.adk import Agent
from google.adk.tools.transfer_to_agent_tool import TransferToAgentTool
from google.genai import types
from .base_agent import initialize_vertex_ai
from .security_agent import create_security_agent
from .direct_adk_agent import create_direct_security_agent
from .hybrid_adk_agent import create_hybrid_security_agent

def create_coordinator_agent(project_id: str) -> Agent:
    """
    Create coordinator agent that delegates to specialized sub-agents.
    Follows ADK LLM-driven delegation pattern.
    """
    initialize_vertex_ai()
    
    # Create specialized sub-agents
    security_agent = create_security_agent()
    direct_agent = create_direct_security_agent(project_id)
    hybrid_agent = create_hybrid_security_agent(project_id)
    
    # Create transfer tools for each sub-agent
    transfer_tools = [
        TransferToAgentTool(
            agent_name="security_agent",
            description="Transfer to comprehensive security analysis agent with full tool access"
        ),
        TransferToAgentTool(
            agent_name="direct_agent", 
            description="Transfer to direct GCP API agent for fast, simple queries"
        ),
        TransferToAgentTool(
            agent_name="hybrid_agent",
            description="Transfer to hybrid agent for balanced performance with custom intelligence"
        )
    ]
    
    # Create coordinator agent with sub-agents
    coordinator = Agent(
        model='gemini-2.0-flash-exp',
        name='security_coordinator',
        description='Intelligent security coordinator that delegates tasks to specialized agents',
        instruction=f"""
        You are the Security Coordinator Agent for GCP project: {project_id}
        
        Your role is to intelligently delegate security tasks to the most appropriate specialized agent:
        
        🤖 **AGENT DELEGATION STRATEGY:**
        
        1. **Direct Agent** (`transfer_to_agent(agent_name='direct_agent')`)
           - Use for: Simple, fast queries requiring direct GCP data
           - Examples: "What's my security score?", "List compute instances", "Show IAM policies"
           - Benefits: Maximum speed, zero backend hops
           
        2. **Hybrid Agent** (`transfer_to_agent(agent_name='hybrid_agent')`) 
           - Use for: Complex analysis requiring both GCP data AND custom business logic
           - Examples: "SOC2 compliance check", "Custom security recommendations", "Policy analysis"
           - Benefits: Speed + Intelligence, eliminates proxies but keeps value-add services
           
        3. **Security Agent** (`transfer_to_agent(agent_name='security_agent')`)
           - Use for: Comprehensive security analysis requiring full tool access
           - Examples: "Complete security audit", "API dependency analysis", "Risk propagation"
           - Benefits: Full capabilities, comprehensive analysis
        
        **🧠 DELEGATION DECISION PROCESS:**
        1. Analyze the user's request complexity and requirements
        2. Consider performance vs. capabilities trade-offs
        3. Use `transfer_to_agent()` to delegate to the most appropriate agent
        4. Let the specialized agent handle the request completely
        
        **🎯 DELEGATION RULES:**
        - Simple queries → Direct Agent (fastest)
        - Complex queries with business context → Hybrid Agent (balanced)  
        - Comprehensive analysis → Security Agent (full capabilities)
        - When in doubt, prefer Hybrid Agent for best balance
        
        **IMPORTANT:** Always use `transfer_to_agent()` to delegate - do not try to answer security questions yourself.
        Your job is intelligent routing, not direct security analysis.
        """,
        tools=transfer_tools,
        sub_agents=[security_agent, direct_agent, hybrid_agent],
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                )
            ]
        )
    )
    
    print(f"🎯 Coordinator Agent created for project: {project_id}")
    print(f"   • Sub-agents: {len(coordinator.sub_agents)}")
    print(f"   • Transfer tools: {len(transfer_tools)}")
    print(f"   • Delegation strategy: LLM-driven based on query complexity")
    
    return coordinator

def create_enhanced_coordinator_with_workflow_agents(project_id: str) -> Agent:
    """
    Create enhanced coordinator with workflow-specific sub-agents.
    Advanced delegation pattern for specialized workflows.
    """
    initialize_vertex_ai()
    
    # Create workflow-specific agents
    compliance_agent = create_compliance_specialist_agent(project_id)
    incident_response_agent = create_incident_response_agent(project_id)
    vulnerability_agent = create_vulnerability_assessment_agent(project_id)
    
    transfer_tools = [
        TransferToAgentTool(
            agent_name="compliance_agent",
            description="Transfer to compliance specialist for SOC2, ISO27001, GDPR, HIPAA evaluations"
        ),
        TransferToAgentTool(
            agent_name="incident_response_agent", 
            description="Transfer to incident response agent for security incident handling and forensics"
        ),
        TransferToAgentTool(
            agent_name="vulnerability_agent",
            description="Transfer to vulnerability assessment agent for detailed security scanning"
        )
    ]
    
    coordinator = Agent(
        model='gemini-2.0-flash-exp',
        name='enhanced_security_coordinator',
        description='Advanced security coordinator with workflow-specific agent delegation',
        instruction=f"""
        You are the Enhanced Security Coordinator for specialized security workflows.
        
        **🎯 WORKFLOW-BASED DELEGATION:**
        
        1. **Compliance Queries** → `transfer_to_agent(agent_name='compliance_agent')`
           - Keywords: "SOC2", "ISO27001", "GDPR", "HIPAA", "compliance", "audit", "regulation"
           - Capabilities: Multi-framework compliance evaluation, custom rules, audit preparation
           
        2. **Incident Response** → `transfer_to_agent(agent_name='incident_response_agent')`
           - Keywords: "incident", "breach", "forensics", "threat", "attack", "compromise"
           - Capabilities: Incident investigation, threat hunting, forensic analysis
           
        3. **Vulnerability Assessment** → `transfer_to_agent(agent_name='vulnerability_agent')`
           - Keywords: "vulnerability", "scan", "CVE", "weakness", "exploit", "pentest"
           - Capabilities: Detailed security scanning, vulnerability prioritization, remediation
        
        **🧠 INTELLIGENT ROUTING:**
        - Analyze query intent and keywords
        - Consider workflow complexity
        - Route to most specialized agent
        - Provide context to receiving agent
        
        Project: {project_id}
        Always delegate - never handle security tasks directly.
        """,
        tools=transfer_tools,
        sub_agents=[compliance_agent, incident_response_agent, vulnerability_agent],
        generate_content_config=types.GenerateContentConfig(
            safety_settings=[
                types.SafetySetting(
                    category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=types.HarmBlockThreshold.OFF,
                )
            ]
        )
    )
    
    return coordinator

def create_compliance_specialist_agent(project_id: str) -> Agent:
    """Create specialized compliance agent."""
    from .hybrid_adk_agent import create_hybrid_security_agent
    
    # Base this on hybrid agent but specialize for compliance
    base_agent = create_hybrid_security_agent(project_id)
    
    # Create compliance-focused agent
    compliance_agent = Agent(
        model='gemini-2.0-flash-exp',
        name='compliance_specialist',
        description='Specialized agent for compliance framework evaluation and audit preparation',
        instruction=f"""
        You are a Compliance Specialist Agent focused on security compliance frameworks.
        
        **🎯 COMPLIANCE SPECIALIZATION:**
        - SOC2 Type I & II evaluations
        - ISO27001 compliance assessment  
        - GDPR data protection compliance
        - HIPAA security rule compliance
        - Custom compliance framework evaluation
        
        **🔧 COMPLIANCE METHODOLOGY:**
        1. Use hybrid ADK pattern for optimal performance
        2. Direct GCP API calls for security posture data
        3. Custom compliance services for framework-specific rules
        4. Generate detailed compliance reports with evidence
        5. Provide remediation roadmaps for non-compliance
        
        Project: {project_id}
        Focus on actionable compliance guidance and audit preparation.
        """,
        tools=base_agent.tools,  # Inherit hybrid agent tools
        generate_content_config=base_agent.generate_content_config
    )
    
    return compliance_agent

def create_incident_response_agent(project_id: str) -> Agent:
    """Create specialized incident response agent."""
    from .security_agent import create_security_agent
    
    base_agent = create_security_agent()
    
    incident_agent = Agent(
        model='gemini-2.0-flash-exp',
        name='incident_response_specialist',
        description='Specialized agent for security incident response and forensic analysis',
        instruction=f"""
        You are an Incident Response Specialist Agent for security incident handling.
        
        **🚨 INCIDENT RESPONSE CAPABILITIES:**
        - Threat detection and analysis
        - Incident investigation workflows
        - Digital forensics and evidence collection
        - Impact assessment and containment strategies
        - Post-incident remediation and lessons learned
        
        **🔍 INVESTIGATION METHODOLOGY:**
        1. Rapid threat assessment using all available tools
        2. Evidence collection from GCP logs and configurations
        3. Timeline reconstruction and attack vector analysis
        4. Risk assessment and impact evaluation
        5. Containment and remediation recommendations
        
        Project: {project_id}
        Prioritize rapid response and thorough investigation.
        """,
        tools=base_agent.tools,
        generate_content_config=base_agent.generate_content_config
    )
    
    return incident_agent

def create_vulnerability_assessment_agent(project_id: str) -> Agent:
    """Create specialized vulnerability assessment agent."""
    from .direct_adk_agent import create_direct_security_agent
    
    base_agent = create_direct_security_agent(project_id)
    
    vuln_agent = Agent(
        model='gemini-2.0-flash-exp', 
        name='vulnerability_assessment_specialist',
        description='Specialized agent for vulnerability assessment and security scanning',
        instruction=f"""
        You are a Vulnerability Assessment Specialist focused on security scanning and weakness identification.
        
        **🔍 VULNERABILITY ASSESSMENT FOCUS:**
        - Comprehensive security scanning
        - CVE mapping and risk prioritization  
        - Configuration weakness identification
        - Penetration testing methodology
        - Remediation prioritization and guidance
        
        **⚡ ASSESSMENT METHODOLOGY:**
        1. Use direct ADK pattern for maximum scanning speed
        2. Rapid GCP resource enumeration and configuration review
        3. Security finding correlation and risk scoring
        4. Vulnerability prioritization based on exploitability
        5. Clear remediation guidance with timelines
        
        Project: {project_id}
        Focus on actionable vulnerability findings and remediation priorities.
        """,
        tools=base_agent.tools,
        generate_content_config=base_agent.generate_content_config
    )
    
    return vuln_agent

# Factory functions for different coordinator patterns
def create_basic_coordinator(project_id: str) -> Agent:
    """Create basic coordinator with performance-focused agents."""
    return create_coordinator_agent(project_id)

def create_workflow_coordinator(project_id: str) -> Agent:
    """Create workflow-specialized coordinator.""" 
    return create_enhanced_coordinator_with_workflow_agents(project_id)