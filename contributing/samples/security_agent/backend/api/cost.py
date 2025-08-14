"""
Cost Analysis and FinOps API endpoints
Provides cost optimization recommendations with security context
"""

from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
import logging
import os
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)
router = APIRouter()

# Mock cost data for demonstration
MOCK_COST_DATA = {
    "mgm-digitalconcierge": {
        "current_month_spend": 4523.67,
        "projected_month_spend": 5102.34,
        "last_month_spend": 4234.12,
        "budget": 5000.00,
        "services_breakdown": {
            "Compute Engine": {
                "cost": 1823.45,
                "percentage": 40.3,
                "trend": "increasing",
                "waste_detected": 423.12
            },
            "Cloud Storage": {
                "cost": 567.89,
                "percentage": 12.5,
                "trend": "stable",
                "waste_detected": 89.34
            },
            "BigQuery": {
                "cost": 892.34,
                "percentage": 19.7,
                "trend": "increasing",
                "waste_detected": 0
            },
            "Cloud SQL": {
                "cost": 445.67,
                "percentage": 9.8,
                "trend": "stable",
                "waste_detected": 156.78
            },
            "Network": {
                "cost": 234.56,
                "percentage": 5.2,
                "trend": "decreasing",
                "waste_detected": 45.23
            },
            "Cloud Functions": {
                "cost": 123.45,
                "percentage": 2.7,
                "trend": "stable",
                "waste_detected": 0
            },
            "Other": {
                "cost": 436.31,
                "percentage": 9.6,
                "trend": "stable",
                "waste_detected": 0
            }
        },
        "unused_resources": [
            {
                "type": "Compute Instance",
                "name": "dev-instance-old-1",
                "monthly_cost": 142.34,
                "days_unused": 45,
                "recommendation": "Delete or stop instance"
            },
            {
                "type": "Static IP",
                "name": "unused-ip-1",
                "monthly_cost": 23.45,
                "days_unused": 90,
                "recommendation": "Release unattached IP"
            },
            {
                "type": "Static IP",
                "name": "unused-ip-2",
                "monthly_cost": 23.45,
                "days_unused": 60,
                "recommendation": "Release unattached IP"
            },
            {
                "type": "Persistent Disk",
                "name": "orphaned-disk-1",
                "monthly_cost": 45.67,
                "days_unused": 30,
                "recommendation": "Delete or snapshot and delete"
            },
            {
                "type": "Cloud SQL Instance",
                "name": "test-mysql-instance",
                "monthly_cost": 156.78,
                "days_unused": 20,
                "recommendation": "Delete test instance"
            }
        ],
        "rightsizing_opportunities": [
            {
                "resource": "prod-web-server-1",
                "current_type": "n2-standard-8",
                "recommended_type": "n2-standard-4",
                "current_cost": 245.67,
                "potential_cost": 122.84,
                "monthly_savings": 122.83,
                "cpu_utilization": "12%",
                "memory_utilization": "23%"
            },
            {
                "resource": "prod-app-server-1",
                "current_type": "n2-highmem-4",
                "recommended_type": "n2-standard-4",
                "current_cost": 189.45,
                "potential_cost": 122.84,
                "monthly_savings": 66.61,
                "cpu_utilization": "34%",
                "memory_utilization": "28%"
            }
        ],
        "commitment_recommendations": [
            {
                "type": "Committed Use Discount",
                "resource_type": "CPU",
                "recommended_commitment": "24 vCPUs",
                "term": "1 year",
                "monthly_savings": 234.56,
                "annual_savings": 2814.72
            }
        ],
        "security_related_costs": {
            "unnecessary_public_ips": {
                "count": 5,
                "monthly_cost": 117.25,
                "security_risk": "HIGH"
            },
            "over_provisioned_iam": {
                "unused_service_accounts": 8,
                "recommendation": "Remove unused service accounts"
            },
            "unencrypted_resources": {
                "resources": 3,
                "potential_cmek_cost": 15.00,
                "security_benefit": "HIGH"
            }
        }
    }
}

@router.get("/analyze/{project_id}")
async def analyze_costs(
    project_id: str,
    detailed: bool = Query(False, description="Include detailed analysis"),
    include_security: bool = Query(True, description="Include security-related cost analysis")
):
    """Analyze costs and provide optimization recommendations."""
    
    cost_data = MOCK_COST_DATA.get(project_id, {})
    
    if not cost_data:
        return {
            "success": False,
            "error": f"No cost data found for project {project_id}"
        }
    
    # Calculate total waste
    total_waste = sum(item["monthly_cost"] for item in cost_data["unused_resources"])
    total_waste += sum(opp["monthly_savings"] for opp in cost_data["rightsizing_opportunities"])
    
    # Build immediate actions
    immediate_actions = []
    
    # Unused resources actions
    for resource in cost_data["unused_resources"][:3]:  # Top 3
        if resource["type"] == "Compute Instance":
            immediate_actions.append({
                "action": f"Delete unused instance '{resource['name']}'",
                "command": f"gcloud compute instances delete {resource['name']} --zone=us-central1-a",
                "monthly_savings": f"${resource['monthly_cost']:.2f}",
                "impact": f"Save ${resource['monthly_cost']:.2f}/month, unused for {resource['days_unused']} days"
            })
        elif resource["type"] == "Static IP":
            immediate_actions.append({
                "action": f"Release unattached IP '{resource['name']}'",
                "command": f"gcloud compute addresses delete {resource['name']} --region=us-central1",
                "monthly_savings": f"${resource['monthly_cost']:.2f}",
                "impact": f"Save ${resource['monthly_cost']:.2f}/month, reduce attack surface"
            })
        elif resource["type"] == "Persistent Disk":
            immediate_actions.append({
                "action": f"Delete orphaned disk '{resource['name']}'",
                "command": f"gcloud compute disks delete {resource['name']} --zone=us-central1-a",
                "monthly_savings": f"${resource['monthly_cost']:.2f}",
                "impact": f"Save ${resource['monthly_cost']:.2f}/month"
            })
    
    # Rightsizing actions
    for opp in cost_data["rightsizing_opportunities"][:2]:  # Top 2
        immediate_actions.append({
            "action": f"Rightsize '{opp['resource']}' from {opp['current_type']} to {opp['recommended_type']}",
            "command": f"gcloud compute instances set-machine-type {opp['resource']} --machine-type={opp['recommended_type']}",
            "monthly_savings": f"${opp['monthly_savings']:.2f}",
            "impact": f"CPU usage only {opp['cpu_utilization']}, Memory usage only {opp['memory_utilization']}"
        })
    
    response = {
        "success": True,
        "project_id": project_id,
        "summary": {
            "current_month_spend": f"${cost_data['current_month_spend']:.2f}",
            "projected_month_spend": f"${cost_data['projected_month_spend']:.2f}",
            "budget": f"${cost_data['budget']:.2f}",
            "budget_status": "OVER BUDGET" if cost_data['projected_month_spend'] > cost_data['budget'] else "ON TRACK",
            "total_potential_savings": f"${total_waste:.2f}",
            "savings_percentage": f"{(total_waste / cost_data['current_month_spend'] * 100):.1f}%"
        },
        "top_spending_services": [
            {
                "service": name,
                "cost": f"${data['cost']:.2f}",
                "percentage": f"{data['percentage']:.1f}%",
                "trend": data["trend"],
                "waste": f"${data['waste_detected']:.2f}" if data["waste_detected"] > 0 else None
            }
            for name, data in sorted(
                cost_data["services_breakdown"].items(),
                key=lambda x: x[1]["cost"],
                reverse=True
            )[:5]
        ],
        "immediate_actions": immediate_actions,
        "unused_resources": {
            "total_count": len(cost_data["unused_resources"]),
            "total_monthly_waste": f"${sum(r['monthly_cost'] for r in cost_data['unused_resources']):.2f}",
            "resources": cost_data["unused_resources"][:5]  # Top 5
        },
        "rightsizing_opportunities": {
            "total_monthly_savings": f"${sum(o['monthly_savings'] for o in cost_data['rightsizing_opportunities']):.2f}",
            "opportunities": cost_data["rightsizing_opportunities"]
        }
    }
    
    if include_security:
        sec_costs = cost_data.get("security_related_costs", {})
        response["security_cost_impact"] = {
            "unnecessary_public_ips": {
                "finding": f"{sec_costs['unnecessary_public_ips']['count']} instances with public IPs that don't need them",
                "monthly_cost": f"${sec_costs['unnecessary_public_ips']['monthly_cost']:.2f}",
                "security_risk": sec_costs['unnecessary_public_ips']['security_risk'],
                "recommendation": "Use Cloud NAT instead of public IPs",
                "command": "gcloud compute instances remove-access-config <instance-name>"
            },
            "encryption_upgrade": {
                "finding": f"{sec_costs['unencrypted_resources']['resources']} resources without CMEK encryption",
                "additional_cost": f"${sec_costs['unencrypted_resources']['potential_cmek_cost']:.2f}/month",
                "security_benefit": "HIGH - Better key management and compliance",
                "recommendation": "Enable CMEK for sensitive data"
            }
        }
    
    if detailed:
        response["detailed_analysis"] = {
            "cost_trends": {
                "3_month_average": "$4,287.45",
                "6_month_average": "$3,956.23",
                "trend": "Increasing 8% month-over-month"
            },
            "commitment_recommendations": cost_data["commitment_recommendations"],
            "forecasted_annual_spend": f"${cost_data['projected_month_spend'] * 12:.2f}",
            "optimization_score": "C+ (68/100)",
            "peer_comparison": "Your costs are 23% higher than similar projects"
        }
    
    return response

@router.get("/budget-alerts/{project_id}")
async def get_budget_alerts(project_id: str):
    """Get budget alerts and forecasts."""
    cost_data = MOCK_COST_DATA.get(project_id, {})
    
    budget = cost_data.get("budget", 5000)
    current = cost_data.get("current_month_spend", 0)
    projected = cost_data.get("projected_month_spend", 0)
    
    alerts = []
    
    if projected > budget:
        alerts.append({
            "severity": "CRITICAL",
            "message": f"Projected to exceed budget by ${projected - budget:.2f}",
            "recommendation": "Immediate cost reduction needed"
        })
    elif projected > budget * 0.9:
        alerts.append({
            "severity": "WARNING",
            "message": f"Approaching budget limit (projected {(projected/budget*100):.0f}% of budget)",
            "recommendation": "Review and optimize high-cost services"
        })
    
    return {
        "success": True,
        "budget": f"${budget:.2f}",
        "current_spend": f"${current:.2f}",
        "projected_spend": f"${projected:.2f}",
        "alerts": alerts,
        "days_remaining": 15,  # Mock
        "daily_burn_rate": f"${current / 15:.2f}"
    }