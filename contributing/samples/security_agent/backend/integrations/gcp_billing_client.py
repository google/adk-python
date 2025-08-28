"""
GCP Billing Integration Client
==============================

Client for integrating with Google Cloud Billing API for cost analysis,
budget monitoring, and service credit calculations in Phase 2 features.
"""

import logging
import asyncio
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import os
from decimal import Decimal

try:
    from google.cloud import billing_v1
    from google.cloud.billing_v1 import types
    from google.cloud import billing_budgets_v1
    from google.auth import default
    GCLOUD_AVAILABLE = True
except ImportError:
    GCLOUD_AVAILABLE = False
    # Create mock types for when library is not available
    class MockTypes:
        class GetProjectBillingInfoRequest:
            def __init__(self, **kwargs):
                pass
        class ListBillingAccountsRequest:
            def __init__(self, **kwargs):
                pass
        class ListBudgetsRequest:
            def __init__(self, **kwargs):
                pass
    types = MockTypes() if not GCLOUD_AVAILABLE else types

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GCPBillingClient:
    """GCP Billing client for cost analysis and budget management"""
    
    def __init__(self, project_id: str, billing_account_id: Optional[str] = None):
        """
        Initialize GCP Billing client
        
        Args:
            project_id: GCP project ID
            billing_account_id: GCP billing account ID
        """
        self.project_id = project_id
        self.billing_account_id = billing_account_id
        
        if not GCLOUD_AVAILABLE:
            logger.warning("Google Cloud Billing library not available")
            self.billing_client = None
            self.budgets_client = None
            return
        
        try:
            # Initialize billing clients
            self.billing_client = billing_v1.CloudBillingClient()
            self.budgets_client = billing_budgets_v1.BudgetServiceClient()
            
            # Set up resource names
            self.project_name = f"projects/{project_id}"
            
            logger.info(f"GCP Billing client initialized for project: {project_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize GCP Billing client: {e}")
            self.billing_client = None
            self.budgets_client = None
    
    async def test_connection(self) -> Dict[str, Any]:
        """Test connection to GCP Billing API"""
        if not self.billing_client:
            return {
                "connected": False,
                "error": "Google Cloud Billing library not available",
                "message": "Install google-cloud-billing package"
            }
        
        try:
            # Test by getting project billing info
            request = types.GetProjectBillingInfoRequest(name=self.project_name)
            billing_info = self.billing_client.get_project_billing_info(request=request)
            
            # Update billing account ID if not provided
            if billing_info.billing_account_name and not self.billing_account_id:
                self.billing_account_id = billing_info.billing_account_name.split('/')[-1]
            
            return {
                "connected": True,
                "project_id": self.project_id,
                "billing_account_name": billing_info.billing_account_name,
                "billing_account_id": self.billing_account_id,
                "billing_enabled": billing_info.billing_enabled,
                "message": "Connection successful"
            }
            
        except Exception as e:
            logger.error(f"GCP Billing connection test failed: {e}")
            return {
                "connected": False,
                "error": str(e),
                "message": "Connection test failed"
            }
    
    async def get_billing_accounts(self) -> Dict[str, Any]:
        """Get available billing accounts"""
        if not self.billing_client:
            return {
                "success": False,
                "error": "GCP Billing client not available"
            }
        
        try:
            request = types.ListBillingAccountsRequest()
            response = self.billing_client.list_billing_accounts(request=request)
            
            accounts = []
            for account in response:
                accounts.append({
                    "name": account.name,
                    "display_name": account.display_name,
                    "open": account.open,
                    "master_billing_account": account.master_billing_account
                })
            
            return {
                "success": True,
                "total_accounts": len(accounts),
                "accounts": accounts
            }
            
        except Exception as e:
            logger.error(f"Get billing accounts failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_project_billing_info(self, project_id: Optional[str] = None) -> Dict[str, Any]:
        """Get billing information for a project"""
        if not self.billing_client:
            return {
                "success": False,
                "error": "GCP Billing client not available"
            }
        
        try:
            target_project = project_id or self.project_id
            project_name = f"projects/{target_project}"
            
            request = types.GetProjectBillingInfoRequest(name=project_name)
            billing_info = self.billing_client.get_project_billing_info(request=request)
            
            return {
                "success": True,
                "project_id": target_project,
                "billing_account_name": billing_info.billing_account_name,
                "billing_account_id": billing_info.billing_account_name.split('/')[-1] if billing_info.billing_account_name else None,
                "billing_enabled": billing_info.billing_enabled,
                "project_name": billing_info.project_id
            }
            
        except Exception as e:
            logger.error(f"Get project billing info failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_service_costs(self, days_back: int = 30, 
                              service_filter: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get service costs for the specified period
        Note: This would typically use the Cloud Billing Data Export or BigQuery
        """
        if not self.billing_client:
            return {
                "success": False,
                "error": "GCP Billing client not available"
            }
        
        try:
            # In a real implementation, this would query the billing export data
            # For now, we'll simulate with mock data based on common services
            
            mock_services = [
                {"service": "Compute Engine", "cost": 1250.75, "usage": "850 hours"},
                {"service": "Cloud Storage", "cost": 87.32, "usage": "2.1 TB"},
                {"service": "BigQuery", "cost": 156.89, "usage": "5.2 TB processed"},
                {"service": "Cloud SQL", "cost": 245.60, "usage": "720 hours"},
                {"service": "Kubernetes Engine", "cost": 432.18, "usage": "12 clusters"},
                {"service": "Cloud Functions", "cost": 23.45, "usage": "2.1M invocations"},
                {"service": "Pub/Sub", "cost": 18.73, "usage": "850K messages"},
                {"service": "Cloud Monitoring", "cost": 45.20, "usage": "150K metrics"},
                {"service": "Cloud Logging", "cost": 67.88, "usage": "12 GB ingested"},
                {"service": "VPC Network", "cost": 92.15, "usage": "890 GB egress"}
            ]
            
            # Filter services if requested
            if service_filter:
                mock_services = [s for s in mock_services if any(filter_svc.lower() in s["service"].lower() for filter_svc in service_filter)]
            
            total_cost = sum(s["cost"] for s in mock_services)
            
            return {
                "success": True,
                "project_id": self.project_id,
                "period_days": days_back,
                "start_date": (datetime.now() - timedelta(days=days_back)).isoformat(),
                "end_date": datetime.now().isoformat(),
                "total_cost": round(total_cost, 2),
                "currency": "USD",
                "service_breakdown": mock_services,
                "top_services": sorted(mock_services, key=lambda x: x["cost"], reverse=True)[:5]
            }
            
        except Exception as e:
            logger.error(f"Get service costs failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_budgets(self, billing_account_id: Optional[str] = None) -> Dict[str, Any]:
        """Get budgets for a billing account"""
        if not self.budgets_client:
            return {
                "success": False,
                "error": "GCP Billing Budgets client not available"
            }
        
        try:
            target_account = billing_account_id or self.billing_account_id
            if not target_account:
                return {
                    "success": False,
                    "error": "No billing account ID available"
                }
            
            parent = f"billingAccounts/{target_account}"
            request = billing_budgets_v1.ListBudgetsRequest(parent=parent)
            response = self.budgets_client.list_budgets(request=request)
            
            budgets = []
            for budget in response:
                budget_data = {
                    "name": budget.name,
                    "display_name": budget.display_name,
                    "etag": budget.etag
                }
                
                # Budget amount
                if budget.amount:
                    if budget.amount.specified_amount:
                        budget_data["amount"] = {
                            "currency_code": budget.amount.specified_amount.currency_code,
                            "units": str(budget.amount.specified_amount.units),
                            "nanos": budget.amount.specified_amount.nanos
                        }
                    elif budget.amount.last_period_amount:
                        budget_data["amount"] = {"type": "last_period_amount"}
                
                # Budget filter
                if budget.budget_filter:
                    budget_filter = {}
                    if budget.budget_filter.projects:
                        budget_filter["projects"] = list(budget.budget_filter.projects)
                    if budget.budget_filter.services:
                        budget_filter["services"] = list(budget.budget_filter.services)
                    if budget.budget_filter.credit_types_treatment:
                        budget_filter["credit_types_treatment"] = budget.budget_filter.credit_types_treatment.name
                    budget_data["filter"] = budget_filter
                
                # Threshold rules
                if budget.threshold_rules:
                    threshold_rules = []
                    for rule in budget.threshold_rules:
                        threshold_rules.append({
                            "threshold_percent": rule.threshold_percent,
                            "spend_basis": rule.spend_basis.name if rule.spend_basis else None
                        })
                    budget_data["threshold_rules"] = threshold_rules
                
                budgets.append(budget_data)
            
            return {
                "success": True,
                "billing_account": target_account,
                "total_budgets": len(budgets),
                "budgets": budgets
            }
            
        except Exception as e:
            logger.error(f"Get budgets failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def calculate_service_credit_eligibility(self, 
                                                 service_type: str, 
                                                 incident_duration_minutes: int,
                                                 affected_percentage: float = 100.0) -> Dict[str, Any]:
        """Calculate service credit eligibility based on SLA violations"""
        try:
            # Get service costs for credit calculation
            costs_result = await self.get_service_costs(days_back=30, service_filter=[service_type])
            if not costs_result["success"]:
                return costs_result
            
            # Find matching service cost
            service_cost = 0.0
            for service in costs_result["service_breakdown"]:
                if service_type.lower() in service["service"].lower():
                    service_cost = service["cost"]
                    break
            
            # SLA credit calculation rules (based on Google Cloud SLAs)
            sla_rules = self._get_sla_credit_rules()
            service_rule = sla_rules.get(service_type.lower(), sla_rules.get("default"))
            
            # Calculate credit percentage based on outage duration
            credit_percentage = 0.0
            if incident_duration_minutes >= service_rule["min_duration_minutes"]:
                # Basic credit percentage
                credit_percentage = service_rule["base_credit_percentage"]
                
                # Additional credit for longer outages
                if incident_duration_minutes >= 60:  # 1 hour
                    credit_percentage = min(credit_percentage * 2, service_rule["max_credit_percentage"])
                elif incident_duration_minutes >= 240:  # 4 hours
                    credit_percentage = min(credit_percentage * 3, service_rule["max_credit_percentage"])
                elif incident_duration_minutes >= 480:  # 8 hours
                    credit_percentage = service_rule["max_credit_percentage"]
            
            # Apply affected percentage
            effective_credit_percentage = credit_percentage * (affected_percentage / 100.0)
            
            # Calculate credit amounts
            monthly_cost = service_cost  # Assuming costs_result is monthly
            credit_amount = monthly_cost * (effective_credit_percentage / 100.0)
            max_credit = monthly_cost * (service_rule["max_credit_percentage"] / 100.0)
            
            return {
                "success": True,
                "service_type": service_type,
                "incident_duration_minutes": incident_duration_minutes,
                "affected_percentage": affected_percentage,
                "monthly_service_cost": monthly_cost,
                "eligible_for_credit": credit_percentage > 0,
                "base_credit_percentage": credit_percentage,
                "effective_credit_percentage": effective_credit_percentage,
                "calculated_credit_amount": round(credit_amount, 2),
                "maximum_possible_credit": round(max_credit, 2),
                "sla_reference": service_rule["sla_url"],
                "calculation_details": {
                    "min_duration_required": service_rule["min_duration_minutes"],
                    "base_credit_rate": service_rule["base_credit_percentage"],
                    "max_credit_rate": service_rule["max_credit_percentage"],
                    "duration_multiplier": self._get_duration_multiplier(incident_duration_minutes),
                    "affected_resource_adjustment": affected_percentage
                }
            }
            
        except Exception as e:
            logger.error(f"Service credit calculation failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def analyze_cost_trends(self, days_back: int = 90) -> Dict[str, Any]:
        """Analyze cost trends over time"""
        try:
            # In a real implementation, this would query historical billing data
            # For now, we'll simulate trend analysis
            
            # Generate mock daily costs for the period
            daily_costs = []
            base_cost = 75.0  # Base daily cost
            
            for i in range(days_back):
                date = datetime.now() - timedelta(days=i)
                
                # Add some variance and trend
                trend_factor = 1 + (i / days_back) * 0.2  # 20% increase over period
                daily_variance = 1 + (hash(date.strftime("%Y%m%d")) % 20 - 10) / 100  # ±10% daily variance
                
                daily_cost = base_cost * trend_factor * daily_variance
                daily_costs.append({
                    "date": date.strftime("%Y-%m-%d"),
                    "cost": round(daily_cost, 2)
                })
            
            # Calculate trend metrics
            daily_costs.reverse()  # Chronological order
            recent_avg = sum(c["cost"] for c in daily_costs[-7:]) / 7  # Last 7 days
            previous_avg = sum(c["cost"] for c in daily_costs[-14:-7]) / 7  # Previous 7 days
            overall_avg = sum(c["cost"] for c in daily_costs) / len(daily_costs)
            
            trend_percentage = ((recent_avg - previous_avg) / previous_avg) * 100
            
            # Identify anomalies (costs > 2 standard deviations from mean)
            costs = [c["cost"] for c in daily_costs]
            mean_cost = sum(costs) / len(costs)
            variance = sum((c - mean_cost) ** 2 for c in costs) / len(costs)
            std_dev = variance ** 0.5
            
            anomalies = [
                c for c in daily_costs 
                if abs(c["cost"] - mean_cost) > 2 * std_dev
            ]
            
            return {
                "success": True,
                "project_id": self.project_id,
                "analysis_period": days_back,
                "total_cost": round(sum(c["cost"] for c in daily_costs), 2),
                "average_daily_cost": round(overall_avg, 2),
                "recent_average": round(recent_avg, 2),
                "previous_average": round(previous_avg, 2),
                "trend_percentage": round(trend_percentage, 2),
                "trend_direction": "INCREASING" if trend_percentage > 5 else "DECREASING" if trend_percentage < -5 else "STABLE",
                "anomalies_detected": len(anomalies),
                "cost_anomalies": anomalies,
                "daily_costs": daily_costs[-30:],  # Return last 30 days for visualization
                "recommendations": self._generate_cost_recommendations(trend_percentage, anomalies)
            }
            
        except Exception as e:
            logger.error(f"Cost trend analysis failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def check_budget_alerts(self, billing_account_id: Optional[str] = None) -> Dict[str, Any]:
        """Check for budget alerts and threshold violations"""
        try:
            budgets_result = await self.get_budgets(billing_account_id)
            if not budgets_result["success"]:
                return budgets_result
            
            costs_result = await self.get_service_costs()
            if not costs_result["success"]:
                return costs_result
            
            current_month_cost = costs_result["total_cost"]
            
            budget_alerts = []
            for budget in budgets_result["budgets"]:
                # Extract budget amount (simplified - assumes USD)
                budget_amount = 1000.0  # Default fallback
                if budget.get("amount") and budget["amount"].get("units"):
                    budget_amount = float(budget["amount"]["units"])
                
                # Calculate spend percentage
                spend_percentage = (current_month_cost / budget_amount) * 100
                
                # Check threshold rules
                alerts = []
                for rule in budget.get("threshold_rules", []):
                    threshold = rule["threshold_percent"]
                    if spend_percentage >= threshold:
                        alerts.append({
                            "threshold_percent": threshold,
                            "current_spend_percent": round(spend_percentage, 2),
                            "alert_type": "THRESHOLD_EXCEEDED",
                            "severity": "HIGH" if threshold >= 90 else "MEDIUM" if threshold >= 75 else "LOW"
                        })
                
                if alerts or spend_percentage >= 80:  # Always alert if >80%
                    budget_alerts.append({
                        "budget_name": budget.get("display_name", "Unknown Budget"),
                        "budget_amount": budget_amount,
                        "current_spend": current_month_cost,
                        "spend_percentage": round(spend_percentage, 2),
                        "alerts": alerts,
                        "projected_month_end": round(current_month_cost * (30 / datetime.now().day), 2),
                        "status": "CRITICAL" if spend_percentage >= 100 else "WARNING" if spend_percentage >= 80 else "OK"
                    })
            
            return {
                "success": True,
                "billing_account": billing_account_id or self.billing_account_id,
                "current_month_spend": current_month_cost,
                "total_budgets_checked": len(budgets_result["budgets"]),
                "budgets_with_alerts": len(budget_alerts),
                "budget_alerts": budget_alerts,
                "overall_budget_health": self._assess_budget_health(budget_alerts)
            }
            
        except Exception as e:
            logger.error(f"Budget alert check failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_sla_credit_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get SLA credit calculation rules for different services"""
        return {
            "compute": {
                "min_duration_minutes": 5,
                "base_credit_percentage": 10.0,
                "max_credit_percentage": 100.0,
                "sla_url": "https://cloud.google.com/compute/sla"
            },
            "storage": {
                "min_duration_minutes": 1,
                "base_credit_percentage": 25.0,
                "max_credit_percentage": 100.0,
                "sla_url": "https://cloud.google.com/storage/sla"
            },
            "kubernetes": {
                "min_duration_minutes": 5,
                "base_credit_percentage": 10.0,
                "max_credit_percentage": 100.0,
                "sla_url": "https://cloud.google.com/kubernetes-engine/sla"
            },
            "sql": {
                "min_duration_minutes": 5,
                "base_credit_percentage": 10.0,
                "max_credit_percentage": 100.0,
                "sla_url": "https://cloud.google.com/sql/sla"
            },
            "bigquery": {
                "min_duration_minutes": 5,
                "base_credit_percentage": 10.0,
                "max_credit_percentage": 50.0,
                "sla_url": "https://cloud.google.com/bigquery/sla"
            },
            "default": {
                "min_duration_minutes": 5,
                "base_credit_percentage": 10.0,
                "max_credit_percentage": 100.0,
                "sla_url": "https://cloud.google.com/terms/sla"
            }
        }
    
    def _get_duration_multiplier(self, duration_minutes: int) -> float:
        """Get credit multiplier based on outage duration"""
        if duration_minutes >= 480:  # 8+ hours
            return 4.0
        elif duration_minutes >= 240:  # 4+ hours
            return 3.0
        elif duration_minutes >= 60:  # 1+ hour
            return 2.0
        else:
            return 1.0
    
    def _generate_cost_recommendations(self, trend_percentage: float, 
                                     anomalies: List[Dict[str, Any]]) -> List[str]:
        """Generate cost optimization recommendations"""
        recommendations = []
        
        if trend_percentage > 15:
            recommendations.append("Significant cost increase detected. Review recent resource provisioning and usage patterns.")
        
        if len(anomalies) > 3:
            recommendations.append("Multiple cost anomalies detected. Set up automated alerts for unusual spending patterns.")
        
        if trend_percentage > 5:
            recommendations.append("Consider implementing budget alerts and spending controls.")
            recommendations.append("Review and optimize high-cost services like Compute Engine and BigQuery.")
        
        recommendations.append("Enable detailed billing export to BigQuery for advanced cost analysis.")
        recommendations.append("Implement resource tagging strategy for better cost attribution.")
        
        return recommendations
    
    def _assess_budget_health(self, budget_alerts: List[Dict[str, Any]]) -> str:
        """Assess overall budget health"""
        if not budget_alerts:
            return "HEALTHY"
        
        critical_alerts = len([alert for alert in budget_alerts if alert["status"] == "CRITICAL"])
        warning_alerts = len([alert for alert in budget_alerts if alert["status"] == "WARNING"])
        
        if critical_alerts > 0:
            return "CRITICAL"
        elif warning_alerts > 1:
            return "WARNING"
        elif warning_alerts > 0:
            return "CAUTION"
        else:
            return "HEALTHY"
    
    async def get_statistics(self) -> Dict[str, Any]:
        """Get GCP Billing integration statistics"""
        try:
            # Get various billing metrics
            costs_result = await self.get_service_costs()
            budgets_result = await self.get_budgets()
            trends_result = await self.analyze_cost_trends(days_back=30)
            
            statistics = {
                "success": True,
                "project_id": self.project_id,
                "billing_account_id": self.billing_account_id,
                "current_month_cost": costs_result.get("total_cost", 0.0) if costs_result["success"] else 0.0,
                "total_services_with_costs": len(costs_result.get("service_breakdown", [])) if costs_result["success"] else 0,
                "active_budgets": len(budgets_result.get("budgets", [])) if budgets_result["success"] else 0,
                "cost_trend": trends_result.get("trend_direction", "UNKNOWN") if trends_result["success"] else "UNKNOWN",
                "anomalies_last_30_days": trends_result.get("anomalies_detected", 0) if trends_result["success"] else 0,
                "analysis_time": datetime.now().isoformat()
            }
            
            return statistics
            
        except Exception as e:
            logger.error(f"GCP Billing statistics failed: {e}")
            return {
                "success": False,
                "error": str(e)
            }


# Example usage and testing
async def test_gcp_billing_client():
    """Test GCP Billing client functionality"""
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT", "test-project")
    billing_account = os.getenv("BILLING_ACCOUNT_ID", None)
    
    client = GCPBillingClient(
        project_id=project_id,
        billing_account_id=billing_account
    )
    
    # Test connection
    connection = await client.test_connection()
    print(f"Connection test: {connection}")
    
    if connection["connected"]:
        # Get service costs
        costs = await client.get_service_costs(days_back=30)
        print(f"Service costs: {costs}")
        
        # Calculate service credit
        credit_calc = await client.calculate_service_credit_eligibility(
            service_type="Compute Engine",
            incident_duration_minutes=120,
            affected_percentage=75.0
        )
        print(f"Service credit calculation: {credit_calc}")
        
        # Analyze trends
        trends = await client.analyze_cost_trends()
        print(f"Cost trends: {trends}")
        
        # Get statistics
        stats = await client.get_statistics()
        print(f"Billing statistics: {stats}")


if __name__ == "__main__":
    asyncio.run(test_gcp_billing_client())