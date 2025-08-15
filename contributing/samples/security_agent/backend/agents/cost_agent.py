"""Cost Optimization Agent with ADK tools."""

import logging
from typing import Dict, Any
from backend.agents.base_agent import BaseADKAgent, create_tool, ToolContext

logger = logging.getLogger(__name__)

@create_tool("analyze_costs", "Analyze costs and identify savings using Billing API")
async def analyze_costs(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze costs using real GCP Billing API."""
    try:
        # Use existing cost analysis API
        from backend.api.cost import analyze_costs as api_analyze_costs
        
        # Get real cost data
        result = await api_analyze_costs(project_id, detailed=True, include_security=True)
        
        if result.get("success"):
            summary = result.get("summary", {})
            recommendations = result.get("recommendations", [])
            immediate_actions = result.get("immediate_actions", [])
            
            # Calculate potential savings from immediate actions
            total_savings = 0
            for action in immediate_actions:
                savings_str = action.get("monthly_savings", "$0")
                try:
                    # Extract number from string like "$123.45/month"
                    savings_value = float(savings_str.replace("$", "").replace("/month", "").replace(",", ""))
                    total_savings += savings_value
                except:
                    pass
            
            # Build recommendations list
            action_recommendations = []
            for action in immediate_actions[:5]:  # Top 5 actions
                action_recommendations.append(action.get("action", ""))
            
            # Add general recommendations
            action_recommendations.extend([
                "Enable budget alerts",
                "Use committed use discounts",
                "Implement resource auto-shutdown policies"
            ])
            
            return {
                "success": True,
                "monthly_spend": summary.get("current_month_spend", "$0"),
                "potential_savings": f"${total_savings:.2f}",
                "unused_resources": len(immediate_actions),
                "budget_status": summary.get("budget_status", "No budget set"),
                "recommendations": action_recommendations,
                "source": "real_billing_api"
            }
        else:
            # Enhanced fallback
            logger.warning("Cost API call failed, using fallback")
            return {
                "success": False,
                "monthly_spend": "$0",
                "potential_savings": "$0",
                "unused_resources": 0,
                "recommendations": [
                    "Enable Cloud Billing API",
                    "Link billing account to project",
                    "Grant billing.viewer role to service account",
                    "Set up cost management policies"
                ],
                "error": result.get("error", "Failed to analyze costs"),
                "source": "fallback"
            }
            
    except ImportError:
        logger.error("Cost API module not available")
        return {
            "success": False,
            "error": "Cost API not available",
            "recommendations": ["Configure Cloud Billing API access"],
            "source": "error"
        }
    except Exception as e:
        logger.error(f"Failed to analyze costs: {e}")
        return {
            "success": False,
            "error": str(e),
            "source": "exception"
        }

class CostOptimizationAgent(BaseADKAgent):
    def __init__(self, project_id: str):
        super().__init__(
            name="CostOptimizationAgent",
            project_id=project_id,
            description="Cost optimization specialist",
            tools=[analyze_costs]
        )
    
    def _get_default_instruction(self) -> str:
        return "Analyze costs and identify optimization opportunities."
    
    async def _default_process(self, query: str):
        result = await analyze_costs(self.project_id, self.context)
        return {"success": True, "response": f"Cost Analysis: {result['monthly_spend']}/month, save {result['potential_savings']}", "data": result}