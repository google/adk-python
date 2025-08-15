"""Network Security Agent with ADK tools."""

import logging
from typing import Dict, Any, List
from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

logger = logging.getLogger(__name__)

@create_tool("analyze_network_security", "Analyze network security configuration using Compute API")
async def analyze_network_security(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze network security using real GCP Compute API."""
    try:
        # Use existing network API
        from backend.api.network import analyze_network_security as api_analyze_network
        
        # Get real network analysis
        result = await api_analyze_network(project_id)
        
        if result.get("success"):
            # Extract real data
            firewall_rules = result.get("firewall_rules", [])
            vpcs = result.get("vpcs", [])
            subnets = result.get("subnets", [])
            
            # Analyze open ports from firewall rules
            open_ports = set()
            risky_rules = []
            
            for rule in firewall_rules:
                if rule.get("sourceRanges") == ["0.0.0.0/0"]:
                    risky_rules.append(rule.get("name", "unknown"))
                    # Extract allowed ports
                    for allowed in rule.get("allowed", []):
                        for port in allowed.get("ports", []):
                            open_ports.add(port)
            
            # Generate recommendations based on real data
            recommendations = []
            if "22" in open_ports:
                recommendations.append("Review SSH access - restrict source IPs")
            if len(risky_rules) > 0:
                recommendations.append(f"Review {len(risky_rules)} rules with 0.0.0.0/0 access")
            if len(vpcs) < 1:
                recommendations.append("Create custom VPC for better network isolation")
            recommendations.extend([
                "Enable Cloud Armor for DDoS protection",
                "Implement Private Google Access",
                "Use VPC Service Controls for API security"
            ])
            
            return {
                "success": True,
                "firewall_rules": len(firewall_rules),
                "open_ports": list(open_ports),
                "vpcs": len(vpcs),
                "subnets": len(subnets),
                "risky_rules": len(risky_rules),
                "recommendations": recommendations,
                "source": "real_compute_api"
            }
        else:
            # Fallback if API fails
            logger.warning("Network API call failed, using enhanced fallback")
            return {
                "success": False,
                "firewall_rules": 0,
                "open_ports": [],
                "vpcs": 0,
                "recommendations": [
                    "Enable Compute Engine API",
                    "Grant compute.viewer role to service account",
                    "Check network admin permissions"
                ],
                "error": result.get("error", "Failed to analyze network"),
                "source": "fallback"
            }
            
    except ImportError:
        logger.error("Network API module not available")
        return {
            "success": False,
            "error": "Network API not available",
            "recommendations": ["Configure Compute Engine API access"],
            "source": "error"
        }
    except Exception as e:
        logger.error(f"Failed to analyze network security: {e}")
        return {
            "success": False,
            "error": str(e),
            "source": "exception"
        }

class NetworkSecurityAgent(BaseADKAgent):
    def __init__(self, project_id: str):
        super().__init__(
            name="NetworkSecurityAgent",
            project_id=project_id,
            description="Network security specialist",
            tools=[analyze_network_security]
        )
    
    def _get_default_instruction(self) -> str:
        return "Analyze network security configuration."
    
    async def _default_process(self, query: str):
        result = await analyze_network_security(self.project_id, self.context)
        return {"success": True, "response": f"Network Analysis: {result}", "data": result}