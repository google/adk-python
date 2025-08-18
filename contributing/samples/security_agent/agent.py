"""
GCP Security Agent - Single Agent Architecture
==============================================

A clean, single-agent implementation that leverages all backend APIs
through simple tool wrappers. No swarm or multi-agent code.
"""

from google.adk import Agent
from google.adk.tools import FunctionTool
import os
import logging
import httpx
from typing import Dict, Optional, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:8000')
API_TIMEOUT = int(os.getenv('API_TIMEOUT', '30'))

# Backend API Client
class BackendAPIClient:
    """Client for backend API interactions"""
    
    def __init__(self, base_url: str = BACKEND_API_URL, timeout: int = API_TIMEOUT):
        self.base_url = base_url
        self.timeout = timeout
    
    def call_api(self, endpoint: str, method: str = "GET", data: Dict = None) -> Dict:
        """Make synchronous API call with error handling"""
        try:
            url = f"{self.base_url}{endpoint}"
            with httpx.Client(timeout=self.timeout) as client:
                if method == "GET":
                    response = client.get(url, params=data)
                elif method == "POST":
                    response = client.post(url, json=data)
                else:
                    response = client.request(method, url, json=data)
                
                response.raise_for_status()
                return response.json()
        except Exception as e:
            logger.warning(f"API call failed for {endpoint}: {e}")
            return {"error": str(e), "status": "backend_unavailable"}

# Initialize API client
api_client = BackendAPIClient()

# ============================================================================
# TOOL WRAPPERS - Each maps directly to a backend API endpoint
# ============================================================================

def discover_assets() -> str:
    """Discover and list GCP assets in the project with enhanced security analysis"""
    result = api_client.call_api("/api/v1/assets/list", "POST", {
        "project_id": PROJECT_ID,
        "include_security_context": True,
        "page_size": 200
    })
    
    if "error" in result:
        return f"Error discovering assets: {result['error']}"
    
    output = f"=== Enhanced Asset Discovery for {PROJECT_ID} ===\n\n"
    
    # Display summary statistics
    if result.get('summary'):
        summary = result['summary']
        output += f"Total Assets: {summary.get('total_assets', 0)}\n"
        
        if summary.get('by_risk_level'):
            output += "\nRisk Distribution:\n"
            risk_levels = summary['by_risk_level']
            for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
                count = risk_levels.get(level, 0)
                if count > 0:
                    emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🔵', 'MINIMAL': '🟢'}.get(level, '⚪')
                    output += f"  {emoji} {level}: {count}\n"
        
        if summary.get('security_issues', 0) > 0:
            output += f"\n⚠️ Security Issues Found: {summary['security_issues']}\n"
        
        output += "\nAssets by Type:\n"
        for asset_type, count in list(summary.get('by_type', {}).items())[:10]:
            output += f"  {asset_type}: {count}\n"
        
        if summary.get('by_region'):
            output += "\nAssets by Region:\n"
            for region, count in list(summary.get('by_region', {}).items())[:5]:
                output += f"  {region}: {count}\n"
    
    # Show high-risk assets if any
    if result.get('assets'):
        high_risk_assets = [asset for asset in result['assets'] 
                           if asset.get('security_context', {}).get('risk_score', 0) >= 61]
        
        if high_risk_assets:
            output += f"\n🚨 HIGH-RISK ASSETS ({len(high_risk_assets)}):\n"
            for asset in high_risk_assets[:5]:
                sec_ctx = asset.get('security_context', {})
                cat = asset.get('categorization', {})
                output += f"  • {cat.get('friendly_type', 'Unknown')} in {cat.get('region', 'unknown')}\n"
                output += f"    Risk Score: {sec_ctx.get('risk_score', 0)}/100 ({sec_ctx.get('risk_level', 'UNKNOWN')})\n"
                if sec_ctx.get('risk_factors'):
                    output += f"    Issues: {', '.join(sec_ctx['risk_factors'][:3])}\n"
            
            if len(high_risk_assets) > 5:
                output += f"    ... and {len(high_risk_assets) - 5} more high-risk assets\n"
    
    if result.get('enhanced_features', {}).get('security_analysis'):
        output += "\n✅ Enhanced security analysis enabled\n"
    
    return output


def analyze_security() -> str:
    """Analyze security findings and vulnerabilities"""
    result = api_client.call_api("/api/v1/security/findings", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error analyzing security: {result['error']}"
    
    output = f"=== Security Analysis for {PROJECT_ID} ===\n\n"
    
    findings = result.get('findings', [])
    output += f"Total Findings: {result.get('total', len(findings))}\n\n"
    
    # Group by severity
    severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
    for finding in findings:
        sev = finding.get('severity', 'UNKNOWN')
        if sev in severity_counts:
            severity_counts[sev] += 1
    
    output += "By Severity:\n"
    for sev, count in severity_counts.items():
        if count > 0:
            output += f"  {sev}: {count}\n"
    
    # Show top findings
    if findings:
        output += "\nTop Findings:\n"
        for finding in findings[:5]:
            output += f"  [{finding.get('severity')}] {finding.get('category', 'Unknown')}: {finding.get('resource', 'Unknown resource')}\n"
    
    return output


def analyze_iam() -> str:
    """Analyze IAM policies and permissions"""
    result = api_client.call_api("/api/v1/iam/analyze", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error analyzing IAM: {result['error']}"
    
    output = f"=== IAM Analysis for {PROJECT_ID} ===\n\n"
    
    if result.get('overprivileged_accounts'):
        output += f"Overprivileged Accounts: {len(result['overprivileged_accounts'])}\n"
        for account in result['overprivileged_accounts'][:3]:
            output += f"  - {account}\n"
    
    if result.get('risky_bindings'):
        output += f"\nRisky IAM Bindings: {len(result['risky_bindings'])}\n"
        for binding in result['risky_bindings'][:3]:
            output += f"  - {binding.get('member', 'Unknown')}: {binding.get('role', 'Unknown role')}\n"
    
    if result.get('service_accounts'):
        output += f"\nService Accounts: {len(result['service_accounts'])}\n"
    
    return output


def analyze_storage() -> str:
    """Analyze storage bucket security"""
    result = api_client.call_api("/api/v1/storage/analyze", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error analyzing storage: {result['error']}"
    
    output = f"=== Storage Security Analysis for {PROJECT_ID} ===\n\n"
    
    buckets = result.get('buckets', [])
    output += f"Total Buckets: {len(buckets)}\n"
    
    public_buckets = [b for b in buckets if b.get('is_public')]
    if public_buckets:
        output += f"\n⚠️ PUBLIC BUCKETS: {len(public_buckets)}\n"
        for bucket in public_buckets[:5]:
            output += f"  - {bucket.get('name')}\n"
    
    # Check for encryption
    unencrypted = [b for b in buckets if not b.get('default_encryption')]
    if unencrypted:
        output += f"\n⚠️ Buckets without default encryption: {len(unencrypted)}\n"
    
    return output


def analyze_monitoring() -> str:
    """Analyze monitoring and alerting configuration"""
    result = api_client.call_api("/api/v1/monitoring/analyze", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error analyzing monitoring: {result['error']}"
    
    output = f"=== Monitoring Configuration for {PROJECT_ID} ===\n\n"
    
    output += f"Alert Policies: {result.get('alert_policies_count', 0)}\n"
    output += f"Uptime Checks: {result.get('uptime_checks_count', 0)}\n"
    output += f"Log Metrics: {result.get('log_metrics_count', 0)}\n"
    
    if result.get('missing_alerts'):
        output += f"\n⚠️ Missing Critical Alerts:\n"
        for alert in result['missing_alerts'][:5]:
            output += f"  - {alert}\n"
    
    return output


def analyze_logs() -> str:
    """Analyze logging configuration and recent security events"""
    result = api_client.call_api("/api/v1/logs/analyze", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error analyzing logs: {result['error']}"
    
    output = f"=== Logging Analysis for {PROJECT_ID} ===\n\n"
    
    output += f"Audit Logs Enabled: {result.get('audit_logs_enabled', False)}\n"
    output += f"Log Sinks: {result.get('log_sinks_count', 0)}\n"
    
    if result.get('recent_security_events'):
        output += f"\nRecent Security Events: {len(result['recent_security_events'])}\n"
        for event in result['recent_security_events'][:3]:
            output += f"  - {event.get('timestamp', 'Unknown time')}: {event.get('description', 'Unknown event')}\n"
    
    return output


def check_org_policies() -> str:
    """Check organization policy compliance"""
    result = api_client.call_api("/api/v1/org-policy/check", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error checking org policies: {result['error']}"
    
    output = f"=== Organization Policy Compliance for {PROJECT_ID} ===\n\n"
    
    output += f"Policies Evaluated: {result.get('total_policies', 0)}\n"
    output += f"Compliant: {result.get('compliant_count', 0)}\n"
    output += f"Non-Compliant: {result.get('non_compliant_count', 0)}\n"
    
    if result.get('violations'):
        output += "\nPolicy Violations:\n"
        for violation in result['violations'][:5]:
            output += f"  - {violation.get('policy', 'Unknown')}: {violation.get('reason', 'Unknown reason')}\n"
    
    return output


def analyze_service_usage() -> str:
    """Analyze enabled services and APIs"""
    result = api_client.call_api("/api/v1/services/analyze", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error analyzing services: {result['error']}"
    
    output = f"=== Service Usage Analysis for {PROJECT_ID} ===\n\n"
    
    output += f"Enabled Services: {result.get('enabled_count', 0)}\n"
    
    if result.get('risky_services'):
        output += "\n⚠️ Potentially Risky Services Enabled:\n"
        for service in result['risky_services'][:5]:
            output += f"  - {service}\n"
    
    if result.get('recommended_services'):
        output += "\n✅ Recommended Security Services to Enable:\n"
        for service in result['recommended_services'][:5]:
            output += f"  - {service}\n"
    
    return output


def check_advisory_notifications() -> str:
    """Check for security advisories and notifications"""
    result = api_client.call_api("/api/v1/advisory/check", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error checking advisories: {result['error']}"
    
    output = f"=== Security Advisories for {PROJECT_ID} ===\n\n"
    
    advisories = result.get('advisories', [])
    output += f"Active Advisories: {len(advisories)}\n"
    
    if advisories:
        output += "\nRecent Advisories:\n"
        for advisory in advisories[:5]:
            output += f"  [{advisory.get('severity', 'INFO')}] {advisory.get('title', 'Unknown')}\n"
            output += f"    {advisory.get('description', '')[:100]}...\n"
    
    return output


def manage_api_keys() -> str:
    """Analyze and manage API keys"""
    result = api_client.call_api("/api/v1/keys/analyze", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error analyzing API keys: {result['error']}"
    
    output = f"=== API Key Analysis for {PROJECT_ID} ===\n\n"
    
    output += f"Total API Keys: {result.get('total_keys', 0)}\n"
    
    if result.get('unrestricted_keys'):
        output += f"\n⚠️ Unrestricted Keys: {len(result['unrestricted_keys'])}\n"
        for key in result['unrestricted_keys'][:3]:
            output += f"  - {key.get('name', 'Unknown')}: {key.get('created', 'Unknown date')}\n"
    
    if result.get('unused_keys'):
        output += f"\n🗑️ Unused Keys (consider deletion): {len(result['unused_keys'])}\n"
    
    return output


def get_security_recommendations() -> str:
    """Get prioritized security recommendations"""
    result = api_client.call_api("/api/v1/recommendations/security", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error getting recommendations: {result['error']}"
    
    output = f"=== Security Recommendations for {PROJECT_ID} ===\n\n"
    
    recommendations = result.get('recommendations', [])
    
    # Group by priority
    critical = [r for r in recommendations if r.get('priority') == 'CRITICAL']
    high = [r for r in recommendations if r.get('priority') == 'HIGH']
    medium = [r for r in recommendations if r.get('priority') == 'MEDIUM']
    
    if critical:
        output += "🔴 CRITICAL (Immediate Action Required):\n"
        for rec in critical[:3]:
            output += f"  • {rec.get('action', 'Review security posture')}\n"
            if rec.get('impact'):
                output += f"    Impact: {rec['impact']}\n"
    
    if high:
        output += "\n🟠 HIGH PRIORITY (Within 24 hours):\n"
        for rec in high[:5]:
            output += f"  • {rec.get('action', 'Implement security best practice')}\n"
    
    if medium:
        output += "\n🟡 MEDIUM PRIORITY (Within 1 week):\n"
        for rec in medium[:5]:
            output += f"  • {rec.get('action', 'Enhance security configuration')}\n"
    
    return output


def run_security_focused_scan() -> str:
    """Run a security-focused scan to identify high-risk assets"""
    result = api_client.call_api("/api/v1/assets/security-scan", "POST", {
        "project_id": PROJECT_ID,
        "page_size": 500,
        "include_security_context": True
    })
    
    if "error" in result:
        return f"Error running security scan: {result['error']}"
    
    output = f"=== SECURITY-FOCUSED ASSET SCAN for {PROJECT_ID} ===\n\n"
    
    # Security Summary
    if result.get('security_summary'):
        summary = result['security_summary']
        output += f"Assets Scanned: {summary.get('total_assets_scanned', 0)}\n"
        
        risk_dist = summary.get('risk_distribution', {})
        output += f"🔴 Critical Risk: {risk_dist.get('CRITICAL', 0)}\n"
        output += f"🟠 High Risk: {risk_dist.get('HIGH', 0)}\n"
        output += f"🟡 Medium Risk: {risk_dist.get('MEDIUM', 0)}\n"
        output += f"🔵 Low Risk: {risk_dist.get('LOW', 0)}\n"
        output += f"🟢 Minimal Risk: {risk_dist.get('MINIMAL', 0)}\n"
        
        if summary.get('most_common_issues'):
            output += f"\nMost Common Security Issues:\n"
            for i, issue in enumerate(summary['most_common_issues'][:5], 1):
                output += f"  {i}. {issue}\n"
    
    # High-Risk Assets
    high_risk_assets = result.get('high_risk_assets', [])
    if high_risk_assets:
        output += f"\n🚨 HIGH-RISK ASSETS REQUIRING IMMEDIATE ATTENTION ({len(high_risk_assets)}):\n"
        output += "=" * 60 + "\n"
        
        for asset in high_risk_assets[:10]:  # Show top 10 highest risk
            output += f"\n• {asset.get('friendly_name', 'Unknown')} ({asset.get('region', 'unknown')})\n"
            output += f"  Risk Score: {asset.get('risk_score', 0)}/100 ({asset.get('risk_level', 'UNKNOWN')})\n"
            
            if asset.get('security_issues'):
                output += f"  Issues: {', '.join(asset['security_issues'][:3])}\n"
            
            if asset.get('recommendations'):
                output += f"  Priority Action: {asset['recommendations'][0]}\n"
            
            if asset.get('is_public'):
                output += "  ⚠️ PUBLICLY ACCESSIBLE\n"
            if not asset.get('is_encrypted'):
                output += "  ⚠️ UNENCRYPTED\n"
        
        if len(high_risk_assets) > 10:
            output += f"\n... and {len(high_risk_assets) - 10} more high-risk assets\n"
    
    # Recommendations
    if result.get('recommendations'):
        rec = result['recommendations']
        output += f"\n📋 IMMEDIATE ACTIONS REQUIRED:\n"
        if rec.get('immediate_action_required', 0) > 0:
            output += f"🔴 {rec['immediate_action_required']} assets need IMMEDIATE attention\n"
        if rec.get('review_within_24h', 0) > 0:
            output += f"🟠 {rec['review_within_24h']} assets should be reviewed within 24 hours\n"
        if rec.get('schedule_review', 0) > 0:
            output += f"🟡 {rec['schedule_review']} assets should be scheduled for review\n"
    
    output += f"\nScan completed at: {result.get('scan_timestamp', 'unknown')}\n"
    
    return output


def run_comprehensive_security_scan() -> str:
    """Run a comprehensive security scan using all available tools"""
    output = "=== COMPREHENSIVE SECURITY SCAN ===\n"
    output += f"Project: {PROJECT_ID}\n"
    output += "=" * 50 + "\n\n"
    
    # Start with security-focused scan
    output += run_security_focused_scan() + "\n\n"
    
    # Run additional security checks
    checks = [
        ("Security Command Center", analyze_security),
        ("IAM Analysis", analyze_iam),
        ("Storage Security", analyze_storage),
        ("Monitoring Status", analyze_monitoring),
        ("Logging Configuration", analyze_logs),
        ("API Key Security", manage_api_keys),
    ]
    
    for check_name, check_func in checks:
        output += f"\n{check_name}:\n"
        output += "-" * 30 + "\n"
        try:
            result = check_func()
            # Extract key findings only
            lines = result.split('\n')
            for line in lines[2:8]:  # Skip header, show key points
                if line.strip():
                    output += line + "\n"
        except Exception as e:
            output += f"Error: {e}\n"
    
    # Get recommendations
    output += "\n" + "=" * 50 + "\n"
    output += get_security_recommendations()
    
    return output


# Create the agent with all available tools
agent_instruction = f"""You are a GCP Security Agent for project {PROJECT_ID}.

You have access to comprehensive security analysis tools that connect to backend APIs:

DISCOVERY & INVENTORY:
- discover_assets: Get enhanced inventory of GCP resources with security context and risk scoring
- analyze_service_usage: Check enabled APIs and services

SECURITY ANALYSIS:
- run_security_focused_scan: Perform comprehensive security scan identifying high-risk assets
- analyze_security: Get Security Command Center findings
- analyze_iam: Review IAM policies and permissions
- analyze_storage: Check storage bucket security
- manage_api_keys: Analyze API key usage and restrictions

COMPLIANCE & MONITORING:
- check_org_policies: Verify organization policy compliance
- analyze_monitoring: Review monitoring and alerting setup
- analyze_logs: Check logging configuration and events
- check_advisory_notifications: Get security advisories

RECOMMENDATIONS:
- get_security_recommendations: Get prioritized action items
- run_comprehensive_security_scan: Execute full security assessment with all tools

When users ask about their GCP security:
1. Use the appropriate tool(s) to gather real data
2. Provide specific findings based on the actual project state
3. Offer clear, actionable recommendations
4. Reference specific resources and configurations found

Always be specific and reference actual findings from the tools.
"""

# Create the agent with all tools
agent = Agent(
    name="gcp_security_agent",
    model="gemini-2.0-flash",
    instruction=agent_instruction,
    tools=[
        FunctionTool(discover_assets),
        FunctionTool(run_security_focused_scan),
        FunctionTool(analyze_security),
        FunctionTool(analyze_iam),
        FunctionTool(analyze_storage),
        FunctionTool(analyze_monitoring),
        FunctionTool(analyze_logs),
        FunctionTool(check_org_policies),
        FunctionTool(analyze_service_usage),
        FunctionTool(check_advisory_notifications),
        FunctionTool(manage_api_keys),
        FunctionTool(get_security_recommendations),
        FunctionTool(run_comprehensive_security_scan),
    ]
)

if __name__ == "__main__":
    print(f"GCP Security Agent configured for project: {PROJECT_ID}")
    print(f"Backend API URL: {BACKEND_API_URL}")
    print("\nAvailable Tools:")
    print("  • discover_assets - Enhanced asset inventory with risk scoring")
    print("  • run_security_focused_scan - Security-focused scan for high-risk assets")
    print("  • analyze_security - Security Command Center findings")
    print("  • analyze_iam - IAM analysis")
    print("  • analyze_storage - Storage security")
    print("  • analyze_monitoring - Monitoring config")
    print("  • analyze_logs - Logging analysis")
    print("  • check_org_policies - Policy compliance")
    print("  • analyze_service_usage - Service analysis")
    print("  • check_advisory_notifications - Security advisories")
    print("  • manage_api_keys - API key security")
    print("  • get_security_recommendations - Get recommendations")
    print("  • run_comprehensive_security_scan - Full security assessment")
    print("\nDeploy to Cloud Run:")
    print(f"gcloud run deploy gcp-security-agent \\")
    print(f"  --source . \\")
    print(f"  --port 8080 \\")
    print(f"  --project {PROJECT_ID} \\")
    print(f"  --set-env-vars BACKEND_API_URL={BACKEND_API_URL} \\")
    print(f"  --allow-unauthenticated \\")
    print(f"  --region us-central1")