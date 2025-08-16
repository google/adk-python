"""
Single GCP Security Agent for Cloud Run
========================================

A single agent with tools that can run on Cloud Run (no multi-agent orchestration).
The agent has all the RADAR logic in its instruction and uses tools to get real data.
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import os
import logging
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Get project ID from environment
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')


def discover_and_analyze_resources() -> str:
    """Discover GCP resources and analyze their security posture.
    
    Combines resource discovery with security analysis in one tool call.
    This implements the Recognition and Assessment phases of RADAR.
    """
    try:
        from google.cloud import asset_v1
        from google.cloud import storage
        import google.auth
        
        credentials, project = google.auth.default()
        if not project:
            project = PROJECT_ID
        
        result = {
            "project": project,
            "resources": {},
            "security_findings": [],
            "summary": {}
        }
        
        # Discovery Phase (Recognition)
        try:
            asset_client = asset_v1.AssetServiceClient(credentials=credentials)
            parent = f"projects/{project}"
            
            assets = asset_client.list_assets(
                request={"parent": parent, "page_size": 100}
            )
            
            asset_count = 0
            asset_types = {}
            compute_resources = []
            storage_resources = []
            iam_resources = []
            
            for asset in assets:
                asset_count += 1
                asset_type = asset.asset_type.split('/')[-1]
                asset_name = asset.name.split('/')[-1]
                
                if asset_type not in asset_types:
                    asset_types[asset_type] = 0
                asset_types[asset_type] += 1
                
                # Categorize resources
                if 'compute' in asset.asset_type.lower():
                    compute_resources.append(asset_name)
                elif 'storage' in asset.asset_type.lower() or 'bucket' in asset.asset_type.lower():
                    storage_resources.append(asset_name)
                    # Security check: public buckets
                    result["security_findings"].append({
                        "resource": asset_name,
                        "type": "storage",
                        "severity": "INFO",
                        "finding": f"Review bucket '{asset_name}' for public access"
                    })
                elif 'iam' in asset.asset_type.lower() or 'serviceaccount' in asset.asset_type.lower():
                    iam_resources.append(asset_name)
                    # Security check: default service accounts
                    if 'compute@developer.gserviceaccount.com' in asset_name:
                        result["security_findings"].append({
                            "resource": asset_name,
                            "type": "iam",
                            "severity": "MEDIUM",
                            "finding": "Default compute service account in use"
                        })
            
            result["resources"] = {
                "total": asset_count,
                "by_type": asset_types,
                "compute": compute_resources[:5],  # Limit for readability
                "storage": storage_resources[:5],
                "iam": iam_resources[:5]
            }
            
        except Exception as e:
            logger.warning(f"Asset inventory error: {e}")
            # Fallback to individual APIs
            try:
                storage_client = storage.Client(project=project, credentials=credentials)
                buckets = list(storage_client.list_buckets())
                
                result["resources"]["storage"] = [b.name for b in buckets[:5]]
                result["resources"]["total"] = len(buckets)
                
                for bucket in buckets:
                    result["security_findings"].append({
                        "resource": bucket.name,
                        "type": "storage",
                        "severity": "INFO",
                        "finding": f"Bucket '{bucket.name}' in {bucket.location}"
                    })
                    
            except Exception as storage_error:
                logger.warning(f"Storage API error: {storage_error}")
        
        # Summary
        result["summary"] = {
            "total_resources": result["resources"].get("total", 0),
            "total_findings": len(result["security_findings"]),
            "critical_findings": len([f for f in result["security_findings"] if f["severity"] == "CRITICAL"]),
            "high_findings": len([f for f in result["security_findings"] if f["severity"] == "HIGH"]),
            "medium_findings": len([f for f in result["security_findings"] if f["severity"] == "MEDIUM"])
        }
        
        # Format as readable text
        output = f"=== GCP Security Analysis for {project} ===\n\n"
        output += f"RESOURCES DISCOVERED:\n"
        output += f"- Total: {result['summary']['total_resources']}\n"
        
        if result["resources"].get("compute"):
            output += f"- Compute: {len(result['resources']['compute'])} instances\n"
        if result["resources"].get("storage"):
            output += f"- Storage: {len(result['resources']['storage'])} buckets\n"
        if result["resources"].get("iam"):
            output += f"- IAM: {len(result['resources']['iam'])} resources\n"
        
        output += f"\nSECURITY FINDINGS:\n"
        output += f"- Total: {result['summary']['total_findings']}\n"
        output += f"- Critical: {result['summary']['critical_findings']}\n"
        output += f"- High: {result['summary']['high_findings']}\n"
        output += f"- Medium: {result['summary']['medium_findings']}\n"
        
        if result["security_findings"]:
            output += f"\nTOP FINDINGS:\n"
            for finding in result["security_findings"][:5]:
                output += f"- [{finding['severity']}] {finding['finding']}\n"
        
        return output
        
    except Exception as e:
        return f"Error in analysis: {str(e)}"


def generate_recommendations() -> str:
    """Generate security recommendations based on common GCP best practices.
    
    Implements the Decision and Action phases of RADAR.
    """
    recommendations = """
    === GCP Security Recommendations ===
    
    IMMEDIATE ACTIONS (Decision Phase):
    1. Enable Security Command Center for comprehensive monitoring
    2. Review and restrict public access to storage buckets
    3. Replace default service accounts with custom ones
    4. Enable audit logging for all services
    5. Implement least-privilege IAM policies
    
    IMPLEMENTATION STEPS (Action Phase):
    
    1. STORAGE SECURITY:
       - Run: gsutil iam get gs://BUCKET_NAME to check permissions
       - Remove allUsers and allAuthenticatedUsers bindings
       - Enable uniform bucket-level access
    
    2. IAM HARDENING:
       - Create custom service accounts for each application
       - Use Workload Identity for GKE workloads
       - Implement IAM conditions for time-based access
    
    3. NETWORK SECURITY:
       - Enable VPC Flow Logs
       - Implement Private Google Access
       - Use Cloud Armor for DDoS protection
    
    4. MONITORING:
       - Enable Cloud Asset Inventory API
       - Set up Security Command Center
       - Configure alert policies for suspicious activities
    
    REVIEW CHECKLIST:
    - [ ] All storage buckets reviewed for public access
    - [ ] Custom service accounts created
    - [ ] Audit logs enabled
    - [ ] Security Command Center configured
    - [ ] Alert policies active
    """
    return recommendations


# Create the single agent with comprehensive RADAR logic
root_agent = Agent(
    name="gcp_security_radar",
    model="gemini-2.0-flash",
    instruction=f"""You are a GCP Security Agent implementing the RADAR methodology for project {PROJECT_ID}.

    RADAR METHODOLOGY:
    - Recognition: Discover what resources exist
    - Assessment: Evaluate security posture
    - Decision: Prioritize what to fix
    - Action: Provide specific remediation steps
    - Review: Verify improvements

    When a user asks about their GCP project or security:
    1. First use 'discover_and_analyze_resources' to get real data (Recognition + Assessment)
    2. Based on findings, explain the security posture
    3. Use 'generate_recommendations' to provide actionable steps (Decision + Action)
    4. Guide them through implementation
    5. Suggest follow-up reviews

    Always be specific and reference actual resources found in the project.
    Provide clear, actionable guidance based on GCP best practices.
    """,
    tools=[
        FunctionTool(discover_and_analyze_resources),
        FunctionTool(generate_recommendations)
    ]
)

# Export for ADK runtime on Cloud Run
agent = root_agent

if __name__ == "__main__":
    print(f"GCP Security RADAR Agent configured for project: {PROJECT_ID}")
    print("\nThis single agent implements the full RADAR methodology with tools.")
    print("Deploy to Cloud Run (no Agent Engine required):")
    print(f"\ngcloud run deploy gcp-security-radar \\")
    print(f"  --source . \\")
    print(f"  --port 8080 \\")
    print(f"  --project {PROJECT_ID} \\")
    print(f"  --allow-unauthenticated \\")
    print(f"  --region us-central1")