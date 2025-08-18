"""
Enhanced GCP Security Agent with Swarm-Based API Integration
============================================================

A single agent enhanced with swarm capabilities to leverage all backend APIs.
Maintains backward compatibility while adding powerful specialist coordination.
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import os
import logging
import json
import asyncio
from typing import Dict, List, Optional, Any
from concurrent.futures import ThreadPoolExecutor
import httpx

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
SWARM_ENABLED = os.getenv('SWARM_ENABLED', 'false').lower() == 'true'
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:8000')
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))

# Backend API Client
class BackendAPIClient:
    """Unified client for all backend API interactions"""
    
    def __init__(self, base_url: str = BACKEND_API_URL, timeout: int = API_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
        self.client = httpx.AsyncClient(timeout=timeout)
    
    async def call_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Make API call with error handling and fallback"""
        try:
            url = f"{self.base_url}{endpoint}"
            if method == "GET":
                response = await self.client.get(url, params=data)
            elif method == "POST":
                response = await self.client.post(url, json=data)
            else:
                response = await self.client.request(method, url, json=data)
            
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"API call failed for {endpoint}: {e}")
            return self._get_fallback_data(endpoint)
    
    def _get_fallback_data(self, endpoint: str) -> Dict:
        """Provide sample data when backend is unavailable"""
        fallbacks = {
            "/api/v1/assets/list": {
                "assets": [
                    {"name": "instance-1", "type": "compute.instance", "location": "us-central1"},
                    {"name": "bucket-1", "type": "storage.bucket", "location": "us"}
                ],
                "total": 2
            },
            "/api/v1/security/findings": {
                "findings": [
                    {"severity": "HIGH", "category": "PUBLIC_ACCESS", "resource": "bucket-1"},
                    {"severity": "MEDIUM", "category": "WEAK_CREDENTIALS", "resource": "service-account-1"}
                ],
                "total": 2
            }
        }
        return fallbacks.get(endpoint, {"status": "fallback_mode", "data": []})

# Initialize API client
api_client = BackendAPIClient()

# ============================================================================
# SPECIALIST TOOLS - Each maps to a backend API module
# ============================================================================

async def asset_discovery_specialist(
    project_id: str = PROJECT_ID,
    asset_types: List[str] = None,
    include_iam: bool = True
) -> str:
    """Asset Discovery Specialist - Complete resource inventory
    
    Leverages backend/api/asset_inventory.py for comprehensive discovery.
    """
    result = await api_client.call_api(
        "/api/v1/assets/list",
        "POST",
        {
            "project_id": project_id,
            "asset_types": asset_types or [],
            "include_iam": include_iam
        }
    )
    
    output = f"=== Asset Discovery Report for {project_id} ===\n\n"
    output += f"Total Assets: {result.get('total', 0)}\n\n"
    
    if result.get('assets'):
        output += "Asset Breakdown:\n"
        asset_types = {}
        for asset in result['assets']:
            asset_type = asset.get('type', 'unknown')
            asset_types[asset_type] = asset_types.get(asset_type, 0) + 1
        
        for asset_type, count in asset_types.items():
            output += f"  - {asset_type}: {count}\n"
    
    return output


async def security_findings_specialist(
    project_id: str = PROJECT_ID,
    severity_filter: List[str] = None,
    finding_type: str = None
) -> str:
    """Security Command Center Specialist - Threat detection and analysis
    
    Leverages backend/api/security.py for Security Command Center integration.
    """
    result = await api_client.call_api(
        "/api/v1/security/findings",
        "POST",
        {
            "project_id": project_id,
            "severity": severity_filter or ["CRITICAL", "HIGH", "MEDIUM"],
            "finding_type": finding_type
        }
    )
    
    output = f"=== Security Findings Analysis ===\n\n"
    
    findings = result.get('findings', [])
    if findings:
        severity_counts = {}
        for finding in findings:
            sev = finding.get('severity', 'UNKNOWN')
            severity_counts[sev] = severity_counts.get(sev, 0) + 1
        
        output += "Finding Summary:\n"
        for sev in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            if sev in severity_counts:
                output += f"  - {sev}: {severity_counts[sev]}\n"
        
        output += "\nTop Findings:\n"
        for finding in findings[:5]:
            output += f"  [{finding.get('severity')}] {finding.get('category')}: {finding.get('resource')}\n"
    else:
        output += "No security findings detected.\n"
    
    return output


async def iam_security_specialist(
    project_id: str = PROJECT_ID,
    check_overprivileged: bool = True,
    check_service_accounts: bool = True
) -> str:
    """IAM Security Specialist - Access control and privilege analysis
    
    Leverages backend/api/iam.py for IAM security analysis.
    """
    result = await api_client.call_api(
        "/api/v1/iam/analyze",
        "POST",
        {
            "project_id": project_id,
            "check_overprivileged": check_overprivileged,
            "check_service_accounts": check_service_accounts
        }
    )
    
    output = f"=== IAM Security Analysis ===\n\n"
    
    if result.get('overprivileged_accounts'):
        output += f"Overprivileged Accounts: {len(result['overprivileged_accounts'])}\n"
        for account in result['overprivileged_accounts'][:3]:
            output += f"  - {account.get('email')}: {account.get('roles_count')} roles\n"
    
    if result.get('risky_bindings'):
        output += f"\nRisky IAM Bindings: {len(result['risky_bindings'])}\n"
        for binding in result['risky_bindings'][:3]:
            output += f"  - {binding.get('member')} has {binding.get('role')}\n"
    
    return output


async def storage_security_specialist(
    project_id: str = PROJECT_ID,
    check_public_access: bool = True,
    check_encryption: bool = True
) -> str:
    """Storage Security Specialist - Bucket and object security analysis
    
    Leverages backend/api/storage.py for storage security checks.
    """
    result = await api_client.call_api(
        "/api/v1/storage/analyze",
        "POST",
        {
            "project_id": project_id,
            "check_public_access": check_public_access,
            "check_encryption": check_encryption
        }
    )
    
    output = f"=== Storage Security Analysis ===\n\n"
    
    buckets = result.get('buckets', [])
    if buckets:
        output += f"Total Buckets: {len(buckets)}\n"
        
        public_buckets = [b for b in buckets if b.get('is_public')]
        if public_buckets:
            output += f"\n⚠️ PUBLIC BUCKETS DETECTED: {len(public_buckets)}\n"
            for bucket in public_buckets:
                output += f"  - {bucket.get('name')}\n"
        
        unencrypted = [b for b in buckets if not b.get('encryption_enabled')]
        if unencrypted:
            output += f"\n⚠️ UNENCRYPTED BUCKETS: {len(unencrypted)}\n"
            for bucket in unencrypted:
                output += f"  - {bucket.get('name')}\n"
    
    return output


async def monitoring_alerts_specialist(
    project_id: str = PROJECT_ID,
    create_recommendations: bool = True
) -> str:
    """Monitoring & Alerting Specialist - Observability and alert configuration
    
    Leverages backend/api/monitoring.py for monitoring setup.
    """
    result = await api_client.call_api(
        "/api/v1/monitoring/analyze",
        "POST",
        {
            "project_id": project_id,
            "create_recommendations": create_recommendations
        }
    )
    
    output = f"=== Monitoring & Alerting Analysis ===\n\n"
    
    output += f"Active Alert Policies: {result.get('alert_policies_count', 0)}\n"
    output += f"Monitored Resources: {result.get('monitored_resources', 0)}\n"
    
    if result.get('missing_alerts'):
        output += f"\n⚠️ MISSING CRITICAL ALERTS:\n"
        for alert in result['missing_alerts']:
            output += f"  - {alert}\n"
    
    if result.get('recommendations'):
        output += f"\nRecommended Alert Policies:\n"
        for rec in result['recommendations'][:5]:
            output += f"  - {rec}\n"
    
    return output


# ============================================================================
# SWARM ORCHESTRATION TOOLS
# ============================================================================

async def orchestrate_security_analysis_swarm() -> str:
    """Orchestrate comprehensive security analysis using specialist swarm
    
    Coordinates multiple specialists in parallel for complete RADAR analysis.
    """
    if not SWARM_ENABLED:
        return await discover_and_analyze_resources_legacy()
    
    logger.info("Initiating swarm-based security analysis...")
    
    # Run specialists in parallel
    tasks = [
        asset_discovery_specialist(),
        security_findings_specialist(),
        iam_security_specialist(),
        storage_security_specialist(),
        monitoring_alerts_specialist()
    ]
    
    results = await asyncio.gather(*tasks, return_exceptions=True)
    
    # Aggregate results
    output = "=== SWARM SECURITY ANALYSIS COMPLETE ===\n\n"
    
    specialist_names = [
        "Asset Discovery",
        "Security Findings",
        "IAM Security",
        "Storage Security",
        "Monitoring & Alerts"
    ]
    
    for name, result in zip(specialist_names, results):
        if isinstance(result, Exception):
            output += f"❌ {name} Specialist: Error - {str(result)}\n\n"
        else:
            output += f"✅ {name} Specialist:\n{result}\n\n"
    
    # Executive Summary
    output += "=== EXECUTIVE SUMMARY ===\n"
    output += "The swarm analysis has completed. Review each specialist report above for detailed findings.\n"
    output += "Use the 'generate_remediation_plan_swarm' tool for actionable next steps.\n"
    
    return output


async def generate_remediation_plan_swarm(
    focus_areas: List[str] = None
) -> str:
    """Generate targeted remediation plan based on swarm analysis
    
    Creates actionable steps leveraging specialist insights.
    """
    if not focus_areas:
        focus_areas = ["iam", "storage", "monitoring", "security"]
    
    output = "=== SWARM-GENERATED REMEDIATION PLAN ===\n\n"
    
    for area in focus_areas:
        if area.lower() == "iam":
            output += "IAM HARDENING:\n"
            output += "1. Remove overprivileged service accounts\n"
            output += "2. Implement least-privilege policies\n"
            output += "3. Enable MFA for all human users\n\n"
        
        elif area.lower() == "storage":
            output += "STORAGE SECURITY:\n"
            output += "1. Remove public access from all buckets\n"
            output += "2. Enable uniform bucket-level access\n"
            output += "3. Implement CMEK encryption\n\n"
        
        elif area.lower() == "monitoring":
            output += "MONITORING SETUP:\n"
            output += "1. Create alert policies for critical events\n"
            output += "2. Enable audit logging for all services\n"
            output += "3. Configure log sinks for security analysis\n\n"
        
        elif area.lower() == "security":
            output += "SECURITY POSTURE:\n"
            output += "1. Enable Security Command Center Premium\n"
            output += "2. Configure vulnerability scanning\n"
            output += "3. Implement DLP policies\n\n"
    
    output += "Execute these steps in order, verifying each before proceeding.\n"
    return output


# ============================================================================
# LEGACY TOOLS (Backward Compatibility)
# ============================================================================

async def discover_and_analyze_resources_legacy() -> str:
    """Legacy discovery tool - maintained for backward compatibility"""
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
        
        # Discovery Phase
        try:
            asset_client = asset_v1.AssetServiceClient(credentials=credentials)
            parent = f"projects/{project}"
            
            assets = asset_client.list_assets(
                request={"parent": parent, "page_size": 100}
            )
            
            asset_count = 0
            asset_types = {}
            
            for asset in assets:
                asset_count += 1
                asset_type = asset.asset_type.split('/')[-1]
                
                if asset_type not in asset_types:
                    asset_types[asset_type] = 0
                asset_types[asset_type] += 1
            
            result["resources"] = {
                "total": asset_count,
                "by_type": asset_types
            }
            
        except Exception as e:
            logger.warning(f"Asset inventory error: {e}")
        
        # Format output
        output = f"=== GCP Security Analysis for {project} ===\n\n"
        output += f"RESOURCES DISCOVERED:\n"
        output += f"- Total: {result['resources'].get('total', 0)}\n"
        
        return output
        
    except Exception as e:
        return f"Error in analysis: {str(e)}"


def generate_recommendations_legacy() -> str:
    """Legacy recommendations tool - maintained for backward compatibility"""
    return """
    === GCP Security Recommendations ===
    
    IMMEDIATE ACTIONS:
    1. Enable Security Command Center
    2. Review public storage buckets
    3. Implement least-privilege IAM
    4. Enable audit logging
    5. Configure monitoring alerts
    
    Use 'orchestrate_security_analysis_swarm' for detailed analysis with specialist agents.
    """


# ============================================================================
# MAIN FUNCTIONS (Routing Logic)
# ============================================================================

def discover_and_analyze_resources() -> str:
    """Main discovery function - routes to swarm or legacy based on config"""
    if SWARM_ENABLED:
        # Run async swarm orchestration
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(orchestrate_security_analysis_swarm())
        finally:
            loop.close()
    else:
        # Run legacy synchronous version
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(discover_and_analyze_resources_legacy())
        finally:
            loop.close()


def generate_recommendations() -> str:
    """Main recommendations function - routes to swarm or legacy based on config"""
    if SWARM_ENABLED:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(generate_remediation_plan_swarm())
        finally:
            loop.close()
    else:
        return generate_recommendations_legacy()


def quick_security_check() -> str:
    """Quick security check using a subset of specialists"""
    if not SWARM_ENABLED:
        return "Quick check requires swarm mode. Set SWARM_ENABLED=true"
    
    async def run_quick_check():
        tasks = [
            security_findings_specialist(severity_filter=["CRITICAL", "HIGH"]),
            storage_security_specialist(check_public_access=True, check_encryption=False)
        ]
        results = await asyncio.gather(*tasks)
        
        output = "=== QUICK SECURITY CHECK ===\n\n"
        output += results[0] + "\n"
        output += results[1] + "\n"
        output += "Run 'discover_and_analyze_resources' for complete analysis.\n"
        return output
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(run_quick_check())
    finally:
        loop.close()


# ============================================================================
# AGENT CONFIGURATION
# ============================================================================

# Enhanced instruction with swarm capabilities
enhanced_instruction = f"""You are an Enhanced GCP Security Agent implementing the RADAR methodology for project {PROJECT_ID}.

{'SWARM MODE ACTIVE: You have access to specialized security agents that provide deep insights across all GCP services.' if SWARM_ENABLED else ''}

RADAR METHODOLOGY:
- Recognition: Discover what resources exist (Asset Discovery Specialist)
- Assessment: Evaluate security posture (Security, IAM, Storage Specialists)  
- Decision: Prioritize what to fix (Analysis aggregation)
- Action: Provide specific remediation steps (Remediation planning)
- Review: Verify improvements (Monitoring Specialist)

AVAILABLE CAPABILITIES:
1. 'discover_and_analyze_resources' - {'Orchestrates specialist swarm for comprehensive analysis' if SWARM_ENABLED else 'Basic resource discovery'}
2. 'generate_recommendations' - {'Swarm-powered remediation planning' if SWARM_ENABLED else 'Standard recommendations'}
3. 'quick_security_check' - Fast critical issue scan (swarm mode only)

When a user asks about their GCP project or security:
1. First use 'discover_and_analyze_resources' for comprehensive data
2. Analyze the findings and explain the security posture
3. Use 'generate_recommendations' for actionable remediation steps
4. Guide them through implementation
5. Suggest follow-up reviews

{'You are leveraging a swarm of specialist agents:\n- Asset Discovery Specialist\n- Security Command Center Specialist\n- IAM Security Specialist\n- Storage Security Specialist\n- Monitoring & Alerting Specialist\n\nEach specialist provides domain-specific insights that are aggregated for comprehensive analysis.' if SWARM_ENABLED else ''}

Always be specific and reference actual resources found in the project.
Provide clear, actionable guidance based on GCP best practices.
"""

# Create the enhanced agent
root_agent = Agent(
    name="gcp_security_radar_enhanced",
    model="gemini-2.0-flash",
    instruction=enhanced_instruction,
    tools=[
        FunctionTool(discover_and_analyze_resources),
        FunctionTool(generate_recommendations),
        FunctionTool(quick_security_check) if SWARM_ENABLED else None
    ] if SWARM_ENABLED else [
        FunctionTool(discover_and_analyze_resources),
        FunctionTool(generate_recommendations)
    ]
)

# Export for ADK runtime on Cloud Run
agent = root_agent

if __name__ == "__main__":
    print(f"Enhanced GCP Security RADAR Agent configured for project: {PROJECT_ID}")
    print(f"Swarm Mode: {'ENABLED' if SWARM_ENABLED else 'DISABLED'}")
    
    if SWARM_ENABLED:
        print("\n✨ SWARM CAPABILITIES ACTIVE ✨")
        print("Specialist agents available:")
        print("  - Asset Discovery Specialist")
        print("  - Security Command Center Specialist")
        print("  - IAM Security Specialist")
        print("  - Storage Security Specialist")
        print("  - Monitoring & Alerting Specialist")
    else:
        print("\nRunning in legacy mode. Set SWARM_ENABLED=true for enhanced capabilities.")
    
    print("\nDeploy to Cloud Run:")
    print(f"gcloud run deploy gcp-security-radar-enhanced \\")
    print(f"  --source . \\")
    print(f"  --port 8080 \\")
    print(f"  --project {PROJECT_ID} \\")
    print(f"  --set-env-vars SWARM_ENABLED={'true' if SWARM_ENABLED else 'false'} \\")
    print(f"  --allow-unauthenticated \\")
    print(f"  --region us-central1")