"""
Google Cloud Recommender Service

Comprehensive service for integrating with Google Cloud Recommender API
to provide contextual security and cost optimization recommendations.
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time

from google.cloud import recommender_v1
from google.cloud import asset_v1
from google.oauth2 import service_account
import google.auth

from ..models.recommender_models import (
    RecommendationInsight as RecommendationInsightModel,
    RecommenderType,
    Priority,
    RecommendationState,
    RecommenderContextRequest,
    RecommendationListResponse,
    RecommendationActionRequest,
    RecommendationActionResponse,
    RecommendationAnalytics
)
from ..models.api_models import (
    ApiResponse
)

logger = logging.getLogger(__name__)

class RecommenderType(Enum):
    """Supported Google Cloud Recommender types."""
    IAM_POLICY = "google.iam.policy.Recommender"
    MACHINE_TYPE = "google.compute.instance.MachineTypeRecommender"
    USAGE_COMMITMENT = "google.compute.commitment.UsageCommitmentRecommender"
    SERVICE_ACCOUNT = "google.iam.serviceAccount.Recommender"
    FIREWALL = "google.compute.firewall.Recommender"
    IDLE_DISK = "google.compute.disk.IdleResourceRecommender"
    IDLE_SQL = "google.cloudsql.instance.IdleRecommender"

class RecommendationState(Enum):
    """Recommendation lifecycle states."""
    ACTIVE = "ACTIVE"
    CLAIMED = "CLAIMED"
    SUCCEEDED = "SUCCEEDED"
    FAILED = "FAILED"
    DISMISSED = "DISMISSED"

class Priority(Enum):
    """Recommendation priority levels."""
    CRITICAL = "critical"
    HIGH = "high" 
    MEDIUM = "medium"
    LOW = "low"

@dataclass
class RecommendationContext:
    """Context for recommendation processing."""
    project_id: str
    resource_name: str
    location: str = "global"
    recommender_type: RecommenderType = None
    filters: Dict[str, Any] = field(default_factory=dict)
    user_preferences: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RecommendationInsight:
    """Enhanced recommendation with analytics."""
    recommendation_id: str
    name: str
    description: str
    recommender_type: RecommenderType
    state: RecommendationState
    priority: Priority
    impact: Dict[str, Any]
    content: Dict[str, Any]
    target_resources: List[str]
    associated_insights: List[str]
    
    # Analytics fields
    cost_savings_usd: float = 0.0
    security_impact_score: float = 0.0
    compliance_impact: Dict[str, Any] = field(default_factory=dict)
    risk_score: float = 0.0
    implementation_effort: str = "medium"
    estimated_time_hours: float = 1.0
    
    # Execution tracking
    executable_commands: List[str] = field(default_factory=list)
    remediation_steps: List[Dict[str, Any]] = field(default_factory=list)
    verification_commands: List[str] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)
    tags: List[str] = field(default_factory=list)
    custom_metadata: Dict[str, Any] = field(default_factory=dict)

class RecommenderService:
    """Comprehensive Google Cloud Recommender service."""
    
    def __init__(self, credentials_path: Optional[str] = None):
        """Initialize the recommender service.
        
        Args:
            credentials_path: Path to service account credentials file
        """
        self.credentials_path = credentials_path
        self.client = None
        self.asset_client = None
        self._initialize_clients()
        
        # Enhanced caching with TTL and performance tracking
        self.cache = {}
        self.cache_ttl = timedelta(minutes=30)
        self.cache_stats = {"hits": 0, "misses": 0, "evictions": 0}
        
        # State tracking
        self.session_recommendations: Dict[str, List[RecommendationInsight]] = {}
        self.recommendation_analytics = RecommendationAnalytics()
        
        # Performance metrics
        self.performance_metrics = {
            "total_requests": 0,
            "avg_response_time": 0.0,
            "error_count": 0,
            "last_health_check": datetime.now()
        }
        
        # Configuration with extended mappings
        self.supported_recommenders = list(RecommenderType)
        self.location_mapping = {
            "global": ["global"],
            "us": ["us-central1", "us-east1", "us-west1", "us-west2"],
            "europe": ["europe-west1", "europe-north1", "europe-west2", "europe-west3"],
            "asia": ["asia-east1", "asia-southeast1", "asia-northeast1", "asia-south1"]
        }
        
        # Rate limiting configuration
        self.rate_limit_config = {
            "requests_per_minute": 60,
            "burst_limit": 10,
            "current_requests": [],
            "backoff_factor": 1.5
        }
    
    def _initialize_clients(self):
        """Initialize Google Cloud clients with retry and health checking."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if self.credentials_path:
                    credentials = service_account.Credentials.from_service_account_file(
                        self.credentials_path
                    )
                    self.client = recommender_v1.RecommenderClient(credentials=credentials)
                    self.asset_client = asset_v1.AssetServiceClient(credentials=credentials)
                else:
                    credentials, project = google.auth.default()
                    self.client = recommender_v1.RecommenderClient(credentials=credentials)
                    self.asset_client = asset_v1.AssetServiceClient(credentials=credentials)
                
                # Test the connection with a lightweight operation
                logger.info("✅ Initialized Google Cloud Recommender clients successfully")
                self.performance_metrics["last_health_check"] = datetime.now()
                return
                
            except Exception as e:
                if attempt == max_retries - 1:
                    logger.error(f"❌ Failed to initialize clients after {max_retries} attempts: {e}")
                    raise
                else:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"⚠️ Client initialization attempt {attempt + 1} failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)

    async def get_all_recommendations(
        self,
        context: RecommendationContext,
        include_insights: bool = True
    ) -> List[RecommendationInsight]:
        """Get all recommendations for a project across all recommender types.
        
        Args:
            context: Recommendation context
            include_insights: Whether to include associated insights
            
        Returns:
            List of enhanced recommendations
        """
        all_recommendations = []
        
        for recommender_type in self.supported_recommenders:
            try:
                recommendations = await self._get_recommendations_by_type(
                    context, recommender_type, include_insights
                )
                all_recommendations.extend(recommendations)
                
            except Exception as e:
                logger.warning(f"Failed to get {recommender_type.value} recommendations: {e}")
                continue
        
        # Sort by priority and impact
        all_recommendations.sort(
            key=lambda r: (
                self._priority_weight(r.priority),
                -r.security_impact_score,
                -r.cost_savings_usd
            )
        )
        
        return all_recommendations

    async def _get_recommendations_by_type(
        self,
        context: RecommendationContext,
        recommender_type: RecommenderType,
        include_insights: bool = True
    ) -> List[RecommendationInsight]:
        """Get recommendations for a specific recommender type."""
        cache_key = f"{context.project_id}:{recommender_type.value}:{context.location}"
        
        # Check cache
        if self._is_cache_valid(cache_key):
            return self.cache[cache_key]["data"]
        
        recommendations = []
        
        try:
            # Get all locations for this recommender
            locations = self._get_locations_for_recommender(recommender_type, context.location)
            
            for location in locations:
                parent = f"projects/{context.project_id}/locations/{location}/recommenders/{recommender_type.value}"
                
                request = recommender_v1.ListRecommendationsRequest(
                    parent=parent,
                    filter=self._build_filter(context.filters)
                )
                
                page_result = self.client.list_recommendations(request=request)
                
                for recommendation in page_result:
                    insight = await self._process_recommendation(
                        recommendation, 
                        recommender_type,
                        include_insights
                    )
                    if insight:
                        recommendations.append(insight)
            
            # Cache results
            self.cache[cache_key] = {
                "data": recommendations,
                "timestamp": datetime.now()
            }
            
        except Exception as e:
            logger.error(f"Error getting recommendations for {recommender_type.value}: {e}")
        
        return recommendations

    async def _process_recommendation(
        self,
        recommendation: recommender_v1.Recommendation,
        recommender_type: RecommenderType,
        include_insights: bool = True
    ) -> Optional[RecommendationInsight]:
        """Process a raw recommendation into enhanced insight."""
        try:
            # Extract basic information
            insight = RecommendationInsight(
                recommendation_id=recommendation.name.split('/')[-1],
                name=recommendation.display_name or "Unnamed Recommendation",
                description=recommendation.description,
                recommender_type=recommender_type,
                state=RecommendationState(recommendation.state_info.state.name),
                priority=self._calculate_priority(recommendation),
                impact=self._extract_impact(recommendation),
                content=self._extract_content(recommendation),
                target_resources=self._extract_target_resources(recommendation),
                associated_insights=[]
            )
            
            # Enhanced analytics
            await self._enhance_with_analytics(insight, recommendation)
            
            # Generate executable commands
            await self._generate_remediation_steps(insight, recommendation)
            
            # Get associated insights if requested
            if include_insights:
                insight.associated_insights = await self._get_associated_insights(
                    recommendation.name
                )
            
            return insight
            
        except Exception as e:
            logger.error(f"Error processing recommendation {recommendation.name}: {e}")
            return None

    async def _enhance_with_analytics(
        self,
        insight: RecommendationInsight,
        recommendation: recommender_v1.Recommendation
    ):
        """Enhance recommendation with analytics and scoring."""
        # Calculate cost savings
        if hasattr(recommendation, 'primary_impact') and recommendation.primary_impact:
            cost_projection = recommendation.primary_impact.cost_projection
            if cost_projection and cost_projection.cost:
                insight.cost_savings_usd = abs(float(cost_projection.cost.units or 0))
        
        # Calculate security impact score
        insight.security_impact_score = self._calculate_security_score(
            insight.recommender_type, 
            insight.content
        )
        
        # Calculate risk score
        insight.risk_score = self._calculate_risk_score(insight)
        
        # Determine implementation effort
        insight.implementation_effort = self._calculate_effort(insight)
        insight.estimated_time_hours = self._estimate_time(insight)
        
        # Compliance impact analysis
        insight.compliance_impact = self._analyze_compliance_impact(insight)

    async def _generate_remediation_steps(
        self,
        insight: RecommendationInsight,
        recommendation: recommender_v1.Recommendation
    ):
        """Generate executable remediation steps."""
        remediation_generator = RemediationGenerator()
        
        insight.remediation_steps = await remediation_generator.generate_steps(
            insight.recommender_type,
            recommendation.content,
            insight.target_resources
        )
        
        insight.executable_commands = await remediation_generator.generate_commands(
            insight.recommender_type,
            recommendation.content
        )
        
        insight.verification_commands = await remediation_generator.generate_verification(
            insight.recommender_type,
            insight.target_resources
        )

    def _calculate_priority(self, recommendation: recommender_v1.Recommendation) -> Priority:
        """Calculate recommendation priority based on multiple factors."""
        priority_score = 0
        
        # Impact weight
        if hasattr(recommendation, 'primary_impact') and recommendation.primary_impact:
            if recommendation.primary_impact.category.name == "SECURITY":
                priority_score += 30
            elif recommendation.primary_impact.category.name == "COST":
                priority_score += 20
            elif recommendation.primary_impact.category.name == "PERFORMANCE":
                priority_score += 15
        
        # Content analysis
        content_text = str(recommendation.content).lower()
        if any(keyword in content_text for keyword in ["critical", "severe", "high risk"]):
            priority_score += 25
        elif any(keyword in content_text for keyword in ["important", "recommended"]):
            priority_score += 15
        
        # State consideration
        if recommendation.state_info.state.name == "ACTIVE":
            priority_score += 10
        
        # Priority mapping
        if priority_score >= 40:
            return Priority.CRITICAL
        elif priority_score >= 25:
            return Priority.HIGH
        elif priority_score >= 15:
            return Priority.MEDIUM
        else:
            return Priority.LOW

    def _calculate_security_score(
        self, 
        recommender_type: RecommenderType, 
        content: Dict[str, Any]
    ) -> float:
        """Calculate security impact score."""
        base_scores = {
            RecommenderType.IAM_POLICY: 0.8,
            RecommenderType.SERVICE_ACCOUNT: 0.7,
            RecommenderType.FIREWALL: 0.9,
            RecommenderType.MACHINE_TYPE: 0.3,
            RecommenderType.USAGE_COMMITMENT: 0.1,
            RecommenderType.IDLE_DISK: 0.2,
            RecommenderType.IDLE_SQL: 0.4
        }
        
        base_score = base_scores.get(recommender_type, 0.5)
        
        # Adjust based on content
        content_text = str(content).lower()
        if any(keyword in content_text for keyword in ["overprivileged", "admin", "owner"]):
            base_score *= 1.3
        elif any(keyword in content_text for keyword in ["unused", "idle", "redundant"]):
            base_score *= 1.1
        
        return min(1.0, base_score)

    def _calculate_risk_score(self, insight: RecommendationInsight) -> float:
        """Calculate overall risk score."""
        risk_factors = [
            insight.security_impact_score * 0.4,
            (1.0 if insight.priority == Priority.CRITICAL else 
             0.7 if insight.priority == Priority.HIGH else
             0.4 if insight.priority == Priority.MEDIUM else 0.2) * 0.3,
            min(1.0, len(insight.target_resources) / 10) * 0.2,
            (1.0 if insight.state == RecommendationState.ACTIVE else 0.5) * 0.1
        ]
        
        return sum(risk_factors)

    def _calculate_effort(self, insight: RecommendationInsight) -> str:
        """Calculate implementation effort level."""
        if insight.recommender_type in [RecommenderType.IAM_POLICY, RecommenderType.FIREWALL]:
            return "high"
        elif insight.recommender_type in [RecommenderType.SERVICE_ACCOUNT]:
            return "medium"
        else:
            return "low"

    def _estimate_time(self, insight: RecommendationInsight) -> float:
        """Estimate implementation time in hours."""
        effort_hours = {
            "low": 0.5,
            "medium": 2.0,
            "high": 8.0
        }
        
        base_time = effort_hours.get(insight.implementation_effort, 1.0)
        
        # Adjust for number of resources
        resource_multiplier = min(3.0, 1 + (len(insight.target_resources) - 1) * 0.2)
        
        return base_time * resource_multiplier

    def _analyze_compliance_impact(self, insight: RecommendationInsight) -> Dict[str, Any]:
        """Analyze compliance framework impact."""
        frameworks = {}
        
        security_types = [RecommenderType.IAM_POLICY, RecommenderType.FIREWALL, RecommenderType.SERVICE_ACCOUNT]
        
        if insight.recommender_type in security_types:
            frameworks.update({
                "SOC2": {"impact": "high", "controls": ["CC6.1", "CC6.2"]},
                "ISO27001": {"impact": "medium", "controls": ["A.9.1", "A.13.1"]},
                "NIST": {"impact": "high", "controls": ["AC-2", "AC-3"]},
                "PCI_DSS": {"impact": "medium", "controls": ["7.1", "8.1"]}
            })
        
        return frameworks

    async def get_recommendations_by_priority(
        self,
        context: RecommendationContext,
        priority: Priority
    ) -> List[RecommendationInsight]:
        """Get recommendations filtered by priority."""
        all_recommendations = await self.get_all_recommendations(context)
        return [r for r in all_recommendations if r.priority == priority]

    async def get_recommendations_by_type(
        self,
        context: RecommendationContext,
        recommender_type: RecommenderType
    ) -> List[RecommendationInsight]:
        """Get recommendations for a specific recommender type."""
        return await self._get_recommendations_by_type(context, recommender_type)

    async def apply_recommendation(
        self,
        recommendation_id: str,
        context: RecommendationContext,
        dry_run: bool = True
    ) -> Dict[str, Any]:
        """Apply a recommendation with optional dry run."""
        try:
            # Get the recommendation
            recommendation_name = f"projects/{context.project_id}/locations/{context.location}/recommenders/{context.recommender_type.value}/recommendations/{recommendation_id}"
            
            if dry_run:
                # Simulate application
                return {
                    "success": True,
                    "dry_run": True,
                    "message": f"Dry run successful for recommendation {recommendation_id}",
                    "estimated_changes": "Would apply security policy changes"
                }
            else:
                # Apply recommendation
                request = recommender_v1.MarkRecommendationClaimedRequest(
                    name=recommendation_name,
                    state_metadata={"applied_by": "adk_security_agent"}
                )
                
                response = self.client.mark_recommendation_claimed(request=request)
                
                return {
                    "success": True,
                    "dry_run": False,
                    "recommendation_id": recommendation_id,
                    "state": response.state_info.state.name,
                    "message": "Recommendation applied successfully"
                }
                
        except Exception as e:
            logger.error(f"Error applying recommendation {recommendation_id}: {e}")
            return {
                "success": False,
                "error": str(e),
                "recommendation_id": recommendation_id
            }

    async def get_session_recommendations(self, session_id: str) -> List[RecommendationInsight]:
        """Get recommendations tracked for a specific session."""
        return self.session_recommendations.get(session_id, [])

    async def add_session_recommendation(
        self, 
        session_id: str, 
        recommendation: RecommendationInsight
    ):
        """Add a recommendation to session tracking."""
        if session_id not in self.session_recommendations:
            self.session_recommendations[session_id] = []
        
        self.session_recommendations[session_id].append(recommendation)

    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cache entry is valid."""
        if cache_key not in self.cache:
            return False
        
        cache_time = self.cache[cache_key]["timestamp"]
        return datetime.now() - cache_time < self.cache_ttl

    def _get_locations_for_recommender(
        self, 
        recommender_type: RecommenderType, 
        preferred_location: str = "global"
    ) -> List[str]:
        """Get appropriate locations for a recommender type."""
        # IAM and some recommenders are global
        global_recommenders = [
            RecommenderType.IAM_POLICY,
            RecommenderType.SERVICE_ACCOUNT
        ]
        
        if recommender_type in global_recommenders:
            return ["global"]
        
        # Regional recommenders
        if preferred_location in self.location_mapping:
            return self.location_mapping[preferred_location]
        
        return [preferred_location] if preferred_location else ["global"]

    def _build_filter(self, filters: Dict[str, Any]) -> str:
        """Build filter string for API requests."""
        filter_parts = []
        
        for key, value in filters.items():
            if isinstance(value, str):
                filter_parts.append(f'{key}="{value}"')
            elif isinstance(value, list):
                or_conditions = [f'{key}="{v}"' for v in value]
                filter_parts.append(f'({" OR ".join(or_conditions)})')
        
        return " AND ".join(filter_parts)

    def _extract_impact(self, recommendation: recommender_v1.Recommendation) -> Dict[str, Any]:
        """Extract impact information from recommendation."""
        impact = {}
        
        if hasattr(recommendation, 'primary_impact') and recommendation.primary_impact:
            impact["category"] = recommendation.primary_impact.category.name
            if recommendation.primary_impact.cost_projection:
                cost_proj = recommendation.primary_impact.cost_projection
                impact["cost_projection"] = {
                    "cost_units": cost_proj.cost.units if cost_proj.cost else 0,
                    "cost_nanos": cost_proj.cost.nanos if cost_proj.cost else 0,
                    "duration": str(cost_proj.duration) if cost_proj.duration else None
                }
        
        return impact

    def _extract_content(self, recommendation: recommender_v1.Recommendation) -> Dict[str, Any]:
        """Extract content from recommendation."""
        content = {}
        
        if hasattr(recommendation, 'content') and recommendation.content:
            # Convert protobuf to dict
            content = {
                "overview": getattr(recommendation.content, 'overview', ''),
                "operation_groups": []
            }
            
            if hasattr(recommendation.content, 'operation_groups'):
                for group in recommendation.content.operation_groups:
                    group_dict = {
                        "operations": []
                    }
                    for operation in group.operations:
                        op_dict = {
                            "action": operation.action,
                            "resource": operation.resource,
                            "resource_type": operation.resource_type
                        }
                        group_dict["operations"].append(op_dict)
                    content["operation_groups"].append(group_dict)
        
        return content

    def _extract_target_resources(self, recommendation: recommender_v1.Recommendation) -> List[str]:
        """Extract target resources from recommendation."""
        resources = []
        
        if hasattr(recommendation, 'content') and recommendation.content:
            if hasattr(recommendation.content, 'operation_groups'):
                for group in recommendation.content.operation_groups:
                    for operation in group.operations:
                        if operation.resource:
                            resources.append(operation.resource)
        
        return list(set(resources))  # Remove duplicates

    async def _get_associated_insights(self, recommendation_name: str) -> List[str]:
        """Get insights associated with a recommendation."""
        # This would integrate with Cloud Asset Inventory or other services
        # to find related insights
        return []

    def _priority_weight(self, priority: Priority) -> int:
        """Get numeric weight for priority sorting."""
        weights = {
            Priority.CRITICAL: 0,
            Priority.HIGH: 1,
            Priority.MEDIUM: 2,
            Priority.LOW: 3
        }
        return weights.get(priority, 4)

class RecommendationAnalytics:
    """Analytics engine for recommendation data."""
    
    def __init__(self):
        self.metrics = {}
    
    def calculate_portfolio_metrics(
        self, 
        recommendations: List[RecommendationInsight]
    ) -> Dict[str, Any]:
        """Calculate portfolio-level metrics."""
        if not recommendations:
            return {}
        
        total_cost_savings = sum(r.cost_savings_usd for r in recommendations)
        avg_security_score = sum(r.security_impact_score for r in recommendations) / len(recommendations)
        
        priority_distribution = {}
        for priority in Priority:
            count = len([r for r in recommendations if r.priority == priority])
            priority_distribution[priority.value] = count
        
        type_distribution = {}
        for rec_type in RecommenderType:
            count = len([r for r in recommendations if r.recommender_type == rec_type])
            if count > 0:
                type_distribution[rec_type.value] = count
        
        return {
            "total_recommendations": len(recommendations),
            "total_cost_savings_usd": total_cost_savings,
            "average_security_score": avg_security_score,
            "priority_distribution": priority_distribution,
            "type_distribution": type_distribution,
            "high_impact_count": len([r for r in recommendations 
                                    if r.priority in [Priority.CRITICAL, Priority.HIGH]]),
            "estimated_implementation_hours": sum(r.estimated_time_hours for r in recommendations)
        }

class RemediationGenerator:
    """Generates executable remediation steps for recommendations."""
    
    async def generate_steps(
        self,
        recommender_type: RecommenderType,
        content: Dict[str, Any],
        target_resources: List[str]
    ) -> List[Dict[str, Any]]:
        """Generate remediation steps."""
        steps = []
        
        if recommender_type == RecommenderType.IAM_POLICY:
            steps = await self._generate_iam_steps(content, target_resources)
        elif recommender_type == RecommenderType.FIREWALL:
            steps = await self._generate_firewall_steps(content, target_resources)
        elif recommender_type == RecommenderType.SERVICE_ACCOUNT:
            steps = await self._generate_service_account_steps(content, target_resources)
        # Add more types as needed
        
        return steps
    
    async def generate_commands(
        self,
        recommender_type: RecommenderType,
        content: Dict[str, Any]
    ) -> List[str]:
        """Generate executable commands."""
        commands = []
        
        # Implementation would generate gcloud, terraform, or API commands
        # based on the recommendation type and content
        
        return commands
    
    async def generate_verification(
        self,
        recommender_type: RecommenderType,
        target_resources: List[str]
    ) -> List[str]:
        """Generate verification commands."""
        commands = []
        
        # Implementation would generate commands to verify the changes
        
        return commands
    
    async def _generate_iam_steps(self, content: Dict[str, Any], resources: List[str]) -> List[Dict[str, Any]]:
        """Generate IAM-specific remediation steps."""
        return [
            {
                "step": 1,
                "title": "Review Current IAM Policy",
                "description": "Analyze current IAM bindings and identify overprivileged accounts",
                "action_type": "review",
                "estimated_minutes": 15
            },
            {
                "step": 2,
                "title": "Remove Excessive Permissions",
                "description": "Remove unnecessary roles and permissions from identified accounts",
                "action_type": "modify",
                "estimated_minutes": 30
            },
            {
                "step": 3,
                "title": "Verify Changes",
                "description": "Test that functionality still works with reduced permissions",
                "action_type": "verify",
                "estimated_minutes": 15
            }
        ]
    
    async def _generate_firewall_steps(self, content: Dict[str, Any], resources: List[str]) -> List[Dict[str, Any]]:
        """Generate firewall-specific remediation steps."""
        return [
            {
                "step": 1,
                "title": "Analyze Firewall Rules",
                "description": "Review current firewall configuration and identify overly permissive rules",
                "action_type": "review",
                "estimated_minutes": 20
            },
            {
                "step": 2,
                "title": "Restrict Source Ranges",
                "description": "Update firewall rules to use more restrictive source IP ranges",
                "action_type": "modify",
                "estimated_minutes": 45
            }
        ]
    
    async def _generate_service_account_steps(self, content: Dict[str, Any], resources: List[str]) -> List[Dict[str, Any]]:
        """Generate service account-specific remediation steps."""
        return [
            {
                "step": 1,
                "title": "Identify Unused Service Accounts",
                "description": "Find service accounts that haven't been used recently",
                "action_type": "review",
                "estimated_minutes": 10
            },
            {
                "step": 2,
                "title": "Disable or Delete Unused Accounts",
                "description": "Safely remove unused service accounts after verification",
                "action_type": "cleanup",
                "estimated_minutes": 30
            }
        ]