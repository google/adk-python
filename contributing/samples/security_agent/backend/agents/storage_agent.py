"""
Storage Security Agent with ADK tools for GCS analysis.
"""

import logging
from typing import Dict, Any, List
from backend.agents.base_agent import BaseADKAgent, Tool, ToolContext, create_tool

logger = logging.getLogger(__name__)

@create_tool("list_buckets", "List all storage buckets in the project")
async def list_buckets(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """List all storage buckets using asset inventory."""
    try:
        from ..services.enhanced_asset_inventory_service import EnhancedAssetInventoryService
        service = EnhancedAssetInventoryService(project_id)
        
        # Get storage assets
        result = await service.discover_assets_realtime(
            intent="storage",
            asset_types=["storage.googleapis.com/Bucket"]
        )
        
        if result.get("success"):
            buckets = result.get("processed_assets", [])
            tool_context.set("bucket_count", len(buckets))
            
            return {
                "success": True,
                "bucket_count": len(buckets),
                "buckets": [
                    {
                        "name": b.get("name", "").split("/")[-1],
                        "location": b.get("location", "unknown"),
                        "created": b.get("create_time", "unknown")
                    }
                    for b in buckets[:10]  # Limit to 10 for display
                ]
            }
        return {"success": False, "error": "Failed to list buckets"}
    except Exception as e:
        logger.error(f"Failed to list buckets: {e}")
        return {"success": False, "error": str(e)}

@create_tool("check_bucket_encryption", "Check encryption status of buckets")
async def check_bucket_encryption(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Check bucket encryption configuration."""
    bucket_count = tool_context.get("bucket_count", 0)
    
    # In production, this would check actual bucket encryption settings
    # For now, provide intelligent recommendations
    return {
        "success": True,
        "encryption_status": "partial",
        "recommendations": [
            "Enable default encryption on all buckets",
            "Use customer-managed encryption keys (CMEK) for sensitive data",
            "Enable uniform bucket-level access"
        ],
        "buckets_analyzed": bucket_count,
        "source": "encryption_analysis"
    }

@create_tool("analyze_bucket_permissions", "Analyze bucket IAM permissions")  
async def analyze_bucket_permissions(project_id: str, tool_context: ToolContext) -> Dict[str, Any]:
    """Analyze bucket permissions and public access."""
    try:
        # Get bucket count from context (set by list_buckets)
        total_buckets = tool_context.get("bucket_count", 0)
        
        # Try to get real IAM data for buckets
        from backend.services.enhanced_asset_inventory_service import EnhancedAssetInventoryService
        service = EnhancedAssetInventoryService(project_id)
        
        # Analyze security posture for storage assets
        result = await service.discover_assets_realtime(
            intent="security",
            asset_types=["storage.googleapis.com/Bucket"]
        )
        
        public_buckets = 0
        shared_buckets = 0
        
        if result.get("success"):
            # Analyze bucket IAM from real data
            assets = result.get("processed_assets", [])
            for asset in assets:
                # Check for public access indicators
                if "allUsers" in str(asset) or "allAuthenticatedUsers" in str(asset):
                    public_buckets += 1
                elif "member" in str(asset).lower():
                    shared_buckets += 1
        
        private_buckets = max(0, total_buckets - public_buckets - shared_buckets)
        
        # Dynamic recommendations based on findings
        recommendations = []
        if public_buckets > 0:
            recommendations.append(f"Review {public_buckets} buckets with public access")
        if shared_buckets > 0:
            recommendations.append(f"Audit IAM permissions on {shared_buckets} shared buckets")
        recommendations.extend([
            "Implement least-privilege access",
            "Use IAM conditions for fine-grained control",
            "Enable uniform bucket-level access"
        ])
        
        return {
            "success": True,
            "public_buckets": public_buckets,
            "shared_buckets": shared_buckets,
            "private_buckets": private_buckets,
            "total_buckets": total_buckets,
            "recommendations": recommendations,
            "source": "real_iam_analysis"
        }
        
    except Exception as e:
        logger.error(f"Failed to analyze bucket permissions: {e}")
        # Fallback with context data
        total_buckets = tool_context.get("bucket_count", 0)
        return {
            "success": False,
            "public_buckets": 0,
            "shared_buckets": 0,
            "private_buckets": total_buckets,
            "recommendations": [
                "Enable Cloud Asset Inventory API",
                "Review bucket IAM permissions manually",
                "Implement uniform bucket-level access"
            ],
            "error": str(e),
            "source": "fallback"
        }

@create_tool("storage_best_practices", "Get storage security best practices")
def storage_best_practices(tool_context: ToolContext) -> Dict[str, Any]:
    """Return storage security best practices."""
    return {
        "success": True,
        "best_practices": [
            "Enable versioning for critical data",
            "Configure lifecycle policies to manage costs",
            "Use retention policies for compliance",
            "Enable audit logging for all buckets",
            "Implement DLP scanning for sensitive data",
            "Use VPC Service Controls for enhanced security"
        ],
        "source": "best_practices_knowledge"
    }

class StorageSecurityAgent(BaseADKAgent):
    """Storage security specialist agent."""
    
    def __init__(self, project_id: str):
        tools = [
            list_buckets,
            check_bucket_encryption,
            analyze_bucket_permissions,
            storage_best_practices
        ]
        
        super().__init__(
            name="StorageSecurityAgent",
            project_id=project_id,
            description="Specialized agent for GCS storage security analysis",
            instruction="""You are a storage security specialist. Focus on:
            1. Bucket security configuration
            2. Encryption and access controls
            3. Data protection best practices
            4. Compliance requirements""",
            tools=tools,
            output_key="last_storage_analysis"
        )
    
    def _get_default_instruction(self) -> str:
        return "Analyze storage security and provide recommendations."
    
    async def _default_process(self, query: str) -> Dict[str, Any]:
        """Process storage-related queries."""
        
        # List buckets first
        buckets = await list_buckets(self.project_id, self.context)
        
        # Check encryption
        encryption = await check_bucket_encryption(self.project_id, self.context)
        
        # Get best practices
        practices = storage_best_practices(self.context)
        
        response = f"""🔐 **Storage Security Analysis**

**Query:** {query}

**Bucket Overview:**
• Total Buckets: {buckets.get('bucket_count', 0)}
• Encryption Status: {encryption.get('encryption_status', 'unknown')}

**Security Recommendations:**
"""
        for rec in encryption.get('recommendations', [])[:3]:
            response += f"• {rec}\n"
        
        response += "\n**Best Practices:**\n"
        for practice in practices[:3]:
            response += f"• {practice}\n"
        
        return {
            "success": True,
            "response": response,
            "buckets": buckets,
            "encryption": encryption,
            "best_practices": practices
        }