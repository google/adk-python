"""
Agent Routing Module - Handles intelligent query routing to specialized agents
Extracted from agent_llm.py for better modularity
"""

import logging
from typing import Tuple, Dict, Optional
from enum import Enum

logger = logging.getLogger(__name__)

class AgentType(Enum):
    """Available agent types for routing"""
    RECOMMENDATION = "recommendation"
    SEARCH = "search"
    ASSET_DISCOVERY = "asset_discovery"
    STORAGE = "storage"
    IAM = "iam"
    NETWORK = "network"
    COMPLIANCE = "compliance"
    COST = "cost"
    COORDINATOR = "coordinator"

class QueryRouter:
    """Routes queries to appropriate agents based on intent detection"""
    
    def __init__(self):
        self.routing_patterns = {
            AgentType.RECOMMENDATION: [
                "recommend", "suggestion", "advice", "should i", "what to do",
                "best practice", "optimize", "improve", "fix"
            ],
            AgentType.SEARCH: [
                "search", "find", "lookup", "research", "google", "web search",
                "what is", "how to", "latest", "recent", "news", "documentation", "examples"
            ],
            AgentType.ASSET_DISCOVERY: [
                "resources", "inventory", "assets", "all", "everything", "project",
                "summary", "overview", "compute instances", "virtual machines",
                "databases", "cloud functions", "kubernetes clusters", "gke",
                "what do i have", "show me", "list", "instances", "vms", "functions", "clusters"
            ],
            AgentType.STORAGE: [
                "bucket", "storage", "backup", "archive", "blob", "object"
            ],
            AgentType.IAM: [
                "user", "permission", "iam", "role", "access", "identity", "service account"
            ],
            AgentType.NETWORK: [
                "firewall", "network", "port", "vpc", "subnet", "load balancer", "ip"
            ],
            AgentType.COMPLIANCE: [
                "compliance", "soc2", "gdpr", "iso", "audit", "pci", "hipaa"
            ],
            AgentType.COST: [
                "cost", "spend", "budget", "savings", "optimize", "expensive", "billing"
            ]
        }
    
    def detect_intent(self, query: str, context: Optional[Dict] = None) -> Tuple[AgentType, str, str]:
        """
        Detect query intent and route to appropriate agent
        
        Returns:
            Tuple of (agent_type, agent_name, routing_reason)
        """
        query_lower = query.lower()
        
        # Log query analysis
        logger.info(f"🎯 Analyzing query for routing: '{query[:100]}'")
        
        # Check each agent type's patterns
        for agent_type, patterns in self.routing_patterns.items():
            matched_patterns = [p for p in patterns if p in query_lower]
            if matched_patterns:
                agent_name = self._get_agent_name(agent_type)
                routing_reason = f"{agent_type.value} keywords detected: {matched_patterns}"
                logger.info(f"✅ Routing to {agent_name}: {routing_reason}")
                return agent_type, agent_name, routing_reason
        
        # Default to coordinator
        logger.info("ℹ️ No specific patterns found, using coordinator")
        return AgentType.COORDINATOR, "CoordinatorAgent", "No specific keywords found, using coordinator"
    
    def _get_agent_name(self, agent_type: AgentType) -> str:
        """Get display name for agent type"""
        agent_names = {
            AgentType.RECOMMENDATION: "RecommendationAgent",
            AgentType.SEARCH: "SearchAgent",
            AgentType.ASSET_DISCOVERY: "AssetDiscoveryAgent",
            AgentType.STORAGE: "StorageSecurityAgent",
            AgentType.IAM: "IAMSecurityAgent",
            AgentType.NETWORK: "NetworkSecurityAgent",
            AgentType.COMPLIANCE: "ComplianceAgent",
            AgentType.COST: "CostOptimizationAgent",
            AgentType.COORDINATOR: "CoordinatorAgent"
        }
        return agent_names.get(agent_type, "UnknownAgent")
    
    def get_contextual_suggestions(self, agent_type: AgentType, query: str) -> list:
        """Generate contextual follow-up suggestions based on agent type"""
        
        suggestions_map = {
            AgentType.ASSET_DISCOVERY: [
                "Which assets have security vulnerabilities?",
                "Show me assets created in the last 30 days",
                "What resources are consuming the most cost?",
                "Are there any unused or orphaned resources?",
                "Which assets need encryption enabled?"
            ],
            AgentType.STORAGE: [
                "Which buckets have public access?",
                "Show me buckets without encryption",
                "What's my storage compliance status?",
                "How can I optimize storage costs?",
                "Which buckets have retention policies?"
            ],
            AgentType.IAM: [
                "Who has admin access to this project?",
                "Show me service accounts with excessive permissions",
                "Which users haven't logged in recently?",
                "What are the most risky IAM policies?",
                "How can I implement least privilege?"
            ],
            AgentType.NETWORK: [
                "Which firewall rules allow public access?",
                "Show me resources with external IPs",
                "What VPCs have the weakest security?",
                "Are there any open ports I should close?",
                "How can I improve network segmentation?"
            ],
            AgentType.COMPLIANCE: [
                "What are my SOC2 compliance gaps?",
                "Show me GDPR compliance issues",
                "Which resources need audit logging?",
                "What security controls am I missing?",
                "How can I improve my compliance score?"
            ],
            AgentType.RECOMMENDATION: [
                "What's the easiest recommendation to implement?",
                "Which recommendations have the highest impact?",
                "Show me cost-saving recommendations",
                "What security quick wins can I implement today?",
                "How do I apply these recommendations?"
            ],
            AgentType.COST: [
                "Which resources are the most expensive?",
                "Show me unused resources I can delete",
                "What are my cost optimization opportunities?",
                "How can I reduce my monthly spend?",
                "Which services are growing in cost?"
            ],
            AgentType.SEARCH: [
                "Search for latest security vulnerabilities",
                "Find GCP security best practices",
                "Research recent security incidents",
                "Look up compliance documentation"
            ]
        }
        
        default_suggestions = [
            "What are my highest priority security risks?",
            "Show me resources with public access",
            "Which compliance standards should I focus on?",
            "How can I improve my security posture?"
        ]
        
        return suggestions_map.get(agent_type, default_suggestions)[:5]