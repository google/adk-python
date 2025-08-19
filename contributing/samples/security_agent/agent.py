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
BACKEND_API_URL = os.getenv('BACKEND_API_URL', 'http://localhost:8002')
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
    """ENHANCED: Analyze security findings and vulnerabilities with STORY-002 features"""
    # Use enhanced comprehensive analysis endpoint
    result = api_client.call_api("/api/v1/security/analyze", "POST", {
        "project_id": PROJECT_ID,
        "include_custom_rules": True,
        "include_compliance_check": True,
        "max_findings": 100
    })
    
    if "error" in result:
        # Fallback to basic security findings if enhanced analysis fails
        fallback_result = api_client.call_api("/api/v1/security/findings", "POST", {"project_id": PROJECT_ID})
        if "error" in fallback_result:
            return f"Error analyzing security: {result['error']}"
        return _format_basic_security_analysis(fallback_result)
    
    # Check if enhanced analysis is available
    if result.get('enhanced_features_enabled'):
        return _format_enhanced_security_analysis(result)
    else:
        # Format basic results
        return _format_basic_security_analysis(result)


def analyze_iam() -> str:
    """Enhanced IAM analysis with overprivileged account detection (STORY-003)"""
    result = api_client.call_api("/api/v1/iam/analyze", "GET")
    
    if "error" in result:
        return f"Error analyzing IAM: {result['error']}"
    
    if not result.get("success"):
        return "Failed to analyze IAM configuration"
    
    analysis = result.get("analysis", {})
    project_id = analysis.get("project_id", PROJECT_ID)
    
    output = f"=== 🔐 ENHANCED IAM SECURITY ANALYSIS for {project_id} ===\n\n"
    
    # Security Posture Score
    posture_score = analysis.get("posture_score", 0)
    score_emoji = "🟢" if posture_score >= 80 else "🟡" if posture_score >= 60 else "🔴"
    output += f"🎯 **Security Posture Score**: {score_emoji} {posture_score}/100\n\n"
    
    # Risk Distribution
    risk_dist = analysis.get("risk_distribution", {})
    if risk_dist:
        output += "📊 **Risk Distribution**:\n"
        for level, count in risk_dist.items():
            if count > 0:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "MINIMAL": "🟢"}.get(level, "⚪")
                output += f"  {emoji} {level}: {count}\n"
        output += "\n"
    
    # Key Statistics
    stats = analysis.get("statistics", {})
    output += "📈 **Key Metrics**:\n"
    output += f"  • Service Accounts: {stats.get('service_account_count', 0)}\n"
    output += f"  • Overprivileged Accounts: {stats.get('overprivileged_accounts', 0)}\n"
    output += f"  • Stale Keys (>90 days): {stats.get('stale_keys', 0)}\n"
    output += f"  • Cross-Project Access: {stats.get('cross_project_bindings', 0)}\n"
    output += f"  • External Users: {stats.get('external_users', 0)}\n\n"
    
    # Critical and High-Risk Findings
    findings = analysis.get("findings", [])
    critical_findings = [f for f in findings if f.get("risk_level") == "CRITICAL"]
    high_findings = [f for f in findings if f.get("risk_level") == "HIGH"]
    
    if critical_findings:
        output += "🔴 **CRITICAL FINDINGS** (Immediate Action Required):\n"
        for finding in critical_findings[:3]:
            output += f"  • {finding.get('title', 'Unknown')}\n"
            output += f"    Resource: {finding.get('affected_principal', 'Unknown')}\n"
            if finding.get('remediation_steps'):
                output += f"    → {finding['remediation_steps'][0]}\n"
        output += "\n"
    
    if high_findings:
        output += "🟠 **HIGH-RISK FINDINGS**:\n"
        for finding in high_findings[:5]:
            output += f"  • {finding.get('title', 'Unknown')} (Risk: {finding.get('risk_score', 0)}/100)\n"
            output += f"    {finding.get('description', 'No description')}\n"
        output += "\n"
    
    # Top Recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        output += "💡 **PRIORITY RECOMMENDATIONS**:\n"
        for i, rec in enumerate(recommendations[:7], 1):
            output += f"  {i}. {rec}\n"
        output += "\n"
    
    # Analysis Summary
    total_findings = analysis.get("total_findings", 0)
    critical_count = analysis.get("critical_findings", 0)
    high_count = analysis.get("high_findings", 0)
    
    output += "📋 **EXECUTIVE SUMMARY**:\n"
    if critical_count > 0:
        output += f"🔴 **URGENT**: {critical_count} critical security issues require immediate attention\n"
    if high_count > 0:
        output += f"🟠 **HIGH PRIORITY**: {high_count} high-risk findings need remediation this week\n"
    
    if total_findings == 0:
        output += "✅ **EXCELLENT**: No significant IAM security issues detected\n"
    elif posture_score >= 80:
        output += "✅ **GOOD**: Strong IAM security posture with minor improvements needed\n"
    elif posture_score >= 60:
        output += "⚠️ **FAIR**: IAM security needs attention, focus on high-risk findings\n"
    else:
        output += "❌ **POOR**: Significant IAM security risks require immediate action\n"
    
    # Source and timestamp
    source = result.get("source", "unknown")
    analyzed_at = analysis.get("analyzed_at", analysis.get("timestamp", "Unknown"))
    
    output += f"\n🔍 Analysis Source: {source}\n"
    output += f"⏰ Completed: {analyzed_at}\n"
    
    if source == "enhanced_iam_analyzer":
        output += "\n✨ **Enhanced Analysis Features**:\n"
        output += "  • Overprivileged account detection\n"
        output += "  • Service account key age analysis\n"
        output += "  • Cross-project access monitoring\n"
        output += "  • External user detection\n"
        output += "  • Automated risk scoring (0-100)\n"
    
    return output


def analyze_storage() -> str:
    """Enhanced storage security analysis with comprehensive security checks (STORY-004)"""
    result = api_client.call_api(f"/api/v1/storage/analyze/{PROJECT_ID}", "GET")
    
    if "error" in result:
        return f"Error analyzing storage: {result['error']}"
    
    if not result.get("success"):
        return "Failed to analyze storage configuration"
    
    analysis = result.get("analysis", {})
    project_id = analysis.get("project_id", PROJECT_ID)
    
    output = f"=== 🗄️ ENHANCED STORAGE SECURITY ANALYSIS for {project_id} ===\n\n"
    
    # Security Posture Score
    posture_score = analysis.get("posture_score", 0)
    score_emoji = "🟢" if posture_score >= 80 else "🟡" if posture_score >= 60 else "🔴"
    output += f"🎯 **Storage Security Score**: {score_emoji} {posture_score}/100\n\n"
    
    # Risk Distribution
    risk_dist = analysis.get("risk_distribution", {})
    if risk_dist:
        output += "📊 **Risk Distribution**:\n"
        for level, count in risk_dist.items():
            if count > 0:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵", "MINIMAL": "🟢"}.get(level, "⚪")
                output += f"  {emoji} {level}: {count}\n"
        output += "\n"
    
    # Key Statistics
    stats = analysis.get("statistics", {})
    output += "📈 **Storage Metrics**:\n"
    output += f"  • Total Buckets: {stats.get('total_buckets', 0)}\n"
    output += f"  • Public Buckets: {stats.get('public_buckets', 0)}\n"
    output += f"  • Customer-Encrypted: {stats.get('encrypted_buckets', 0)}\n"
    output += f"  • Compliant Buckets: {stats.get('compliant_buckets', 0)}\n\n"
    
    # Compliance Status
    compliance = analysis.get("compliance_status", {})
    if compliance:
        output += "📋 **Compliance Scores**:\n"
        for framework, score in compliance.items():
            score_emoji = "🟢" if score >= 80 else "🟡" if score >= 60 else "🔴"
            output += f"  {score_emoji} {framework}: {score:.1f}%\n"
        output += "\n"
    
    # Critical and High-Risk Findings
    findings = analysis.get("findings", [])
    critical_findings = [f for f in findings if f.get("risk_level") == "CRITICAL"]
    high_findings = [f for f in findings if f.get("risk_level") == "HIGH"]
    
    if critical_findings:
        output += "🔴 **CRITICAL FINDINGS** (Immediate Action Required):\n"
        for finding in critical_findings[:3]:
            output += f"  • {finding.get('title', 'Unknown')}\n"
            output += f"    Bucket: {finding.get('bucket_name', 'Unknown')}\n"
            if finding.get('remediation_steps'):
                output += f"    → {finding['remediation_steps'][0]}\n"
        output += "\n"
    
    if high_findings:
        output += "🟠 **HIGH-RISK FINDINGS**:\n"
        for finding in high_findings[:5]:
            output += f"  • {finding.get('title', 'Unknown')} (Risk: {finding.get('risk_score', 0)}/100)\n"
            output += f"    Bucket: {finding.get('bucket_name', 'Unknown')}\n"
            compliance_list = finding.get('compliance_frameworks', [])
            if compliance_list:
                output += f"    Compliance: {', '.join(compliance_list[:3])}\n"
        output += "\n"
    
    # Top Recommendations
    recommendations = analysis.get("recommendations", [])
    if recommendations:
        output += "💡 **PRIORITY RECOMMENDATIONS**:\n"
        for i, rec in enumerate(recommendations[:6], 1):
            output += f"  {i}. {rec}\n"
        output += "\n"
    
    # Analysis Summary
    total_buckets = stats.get("total_buckets", 0)
    public_buckets = stats.get("public_buckets", 0)
    encrypted_buckets = stats.get("encrypted_buckets", 0)
    
    output += "📋 **EXECUTIVE SUMMARY**:\n"
    if critical_findings:
        output += f"🔴 **URGENT**: {len(critical_findings)} critical storage security issues require immediate attention\n"
    if public_buckets > 0:
        output += f"🟠 **HIGH PRIORITY**: {public_buckets} public buckets need access control review\n"
    
    if total_buckets == 0:
        output += "ℹ️ **NO BUCKETS**: No storage buckets found in this project\n"
    elif posture_score >= 85:
        output += "✅ **EXCELLENT**: Strong storage security posture with best practices implemented\n"
    elif posture_score >= 70:
        output += "✅ **GOOD**: Solid storage security with minor improvements needed\n"
    elif posture_score >= 50:
        output += "⚠️ **FAIR**: Storage security needs attention, focus on high-risk findings\n"
    else:
        output += "❌ **POOR**: Significant storage security risks require immediate action\n"
    
    # Encryption status
    if total_buckets > 0:
        encryption_rate = (encrypted_buckets / total_buckets) * 100
        if encryption_rate < 50:
            output += f"🔐 **ENCRYPTION**: Only {encryption_rate:.1f}% of buckets use customer-managed encryption\n"
    
    # Source and timestamp
    source = result.get("source", "unknown")
    analyzed_at = analysis.get("analyzed_at", "Unknown")
    
    output += f"\n🔍 Analysis Source: {source}\n"
    output += f"⏰ Completed: {analyzed_at}\n"
    
    if source == "enhanced_storage_analyzer":
        output += "\n✨ **Enhanced Analysis Features**:\n"
        output += "  • Public bucket detection with access type analysis\n"
        output += "  • Customer-managed encryption validation\n"
        output += "  • Lifecycle policy compliance checking\n"
        output += "  • Multi-framework compliance scoring (SOC2, HIPAA, PCI-DSS, GDPR)\n"
        output += "  • Uniform bucket access and public prevention validation\n"
        output += "  • Sensitive data pattern detection\n"
        output += "  • Automated remediation recommendations\n"
    
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
    """Get comprehensive security recommendations with CVSS-based prioritization (STORY-007)"""
    result = api_client.call_api(f"/api/v1/recommendations/enhanced/{PROJECT_ID}")
    
    if "error" in result:
        return f"Error getting enhanced recommendations: {result['error']}"
    
    if not result.get("success"):
        # Fallback to basic recommendations
        return get_basic_security_recommendations()
    
    summary = result.get('summary', {})
    recommendations = result.get('recommendations', [])
    
    output = f"=== Enhanced Security Recommendations for {PROJECT_ID} ===\n\n"
    
    # Summary statistics
    output += f"📊 Summary:\n"
    output += f"  • Total Recommendations: {summary.get('total_recommendations', 0)}\n"
    output += f"  • Critical Issues (P0/P1): {summary.get('critical_count', 0)}\n"
    output += f"  • Estimated Effort: {summary.get('total_estimated_effort_hours', 0):.1f} hours\n"
    output += f"  • Risk Reduction Potential: {summary.get('estimated_risk_reduction', 0):.1f} CVSS points\n"
    
    if summary.get('overdue_count', 0) > 0:
        output += f"  • ⚠️ Overdue Items: {summary['overdue_count']}\n"
    
    output += "\n"
    
    # Group by priority
    p0_recs = [r for r in recommendations if r.get('priority') == 'P0']
    p1_recs = [r for r in recommendations if r.get('priority') == 'P1']
    p2_recs = [r for r in recommendations if r.get('priority') == 'P2']
    
    if p0_recs:
        output += "🔴 P0 CRITICAL (Immediate - within 4 hours):\n"
        for rec in p0_recs[:3]:
            output += f"  • {rec.get('title', 'Critical Security Issue')}\n"
            output += f"    CVSS: {rec.get('cvss_score', 0):.1f} | Business Impact: {rec.get('business_impact', 'Unknown')}\n"
            output += f"    Effort: {rec.get('estimated_effort_hours', 0):.1f}h | Due: {rec.get('due_date', 'ASAP')}\n"
            if rec.get('affected_resources'):
                output += f"    Resources: {', '.join(rec['affected_resources'][:2])}\n"
            output += "\n"
    
    if p1_recs:
        output += "🟠 P1 HIGH (Within 24 hours):\n"
        for rec in p1_recs[:4]:
            output += f"  • {rec.get('title', 'High Priority Issue')}\n"
            output += f"    CVSS: {rec.get('cvss_score', 0):.1f} | Effort: {rec.get('estimated_effort_hours', 0):.1f}h\n"
            if rec.get('automation_script'):
                output += f"    ✅ Automation available\n"
            output += "\n"
    
    if p2_recs:
        output += "🟡 P2 MEDIUM (Within 1 week):\n"
        for rec in p2_recs[:3]:
            output += f"  • {rec.get('title', 'Medium Priority Issue')}\n"
            output += f"    Category: {rec.get('category', 'Security')} | Effort: {rec.get('estimated_effort_hours', 0):.1f}h\n"
    
    # Compliance insights
    if summary.get('by_category'):
        output += "\n📋 By Category:\n"
        for category, count in summary['by_category'].items():
            if count > 0:
                output += f"  • {category}: {count} recommendations\n"
    
    return output


def get_basic_security_recommendations() -> str:
    """Get basic security recommendations (fallback)"""
    result = api_client.call_api("/api/v1/recommendations/live", "POST", {"project_id": PROJECT_ID})
    
    if "error" in result:
        return f"Error getting recommendations: {result['error']}"
    
    output = f"=== Security Recommendations for {PROJECT_ID} ===\n\n"
    
    recommendations = result.get('recommendations', [])
    
    # Group by priority
    critical = [r for r in recommendations if r.get('priority') == 'critical']
    high = [r for r in recommendations if r.get('priority') == 'high']
    medium = [r for r in recommendations if r.get('priority') == 'medium']
    
    if critical:
        output += "🔴 CRITICAL (Immediate Action Required):\n"
        for rec in critical[:3]:
            output += f"  • {rec.get('title', 'Review security posture')}\n"
            if rec.get('impact'):
                output += f"    Impact: {rec['impact']}\n"
    
    if high:
        output += "\n🟠 HIGH PRIORITY (Within 24 hours):\n"
        for rec in high[:5]:
            output += f"  • {rec.get('title', 'Implement security best practice')}\n"
    
    if medium:
        output += "\n🟡 MEDIUM PRIORITY (Within 1 week):\n"
        for rec in medium[:5]:
            output += f"  • {rec.get('title', 'Enhance security configuration')}\n"
    
    return output


def get_priority_recommendations(priority_level: str = "P0") -> str:
    """Get recommendations filtered by priority level"""
    result = api_client.call_api(f"/api/v1/recommendations/priority/{PROJECT_ID}?priority_level={priority_level}")
    
    if "error" in result:
        return f"Error getting {priority_level} recommendations: {result['error']}"
    
    if not result.get("success"):
        return f"Enhanced recommendations not available for priority {priority_level}"
    
    recommendations = result.get('recommendations', [])
    insights = result.get('insights', {})
    
    output = f"=== {priority_level} Priority Recommendations for {PROJECT_ID} ===\n\n"
    
    if insights:
        output += f"⚠️ {insights.get('urgency', 'Action required')}\n"
        output += f"📊 Risk Level: {insights.get('risk_level', 'Unknown')}\n"
        output += f"⏰ Timeline: {insights.get('recommended_action', 'Address promptly')}\n"
        if insights.get('escalation'):
            output += f"📢 Escalation: {insights['escalation']}\n"
        output += "\n"
    
    if not recommendations:
        output += f"✅ No {priority_level} priority recommendations found. Great job!\n"
        return output
    
    for i, rec in enumerate(recommendations[:10], 1):
        output += f"{i}. {rec.get('title', 'Security Issue')}\n"
        output += f"   CVSS Score: {rec.get('cvss_score', 0):.1f}\n"
        output += f"   Business Impact: {rec.get('business_impact', 'Unknown')}\n"
        output += f"   Estimated Effort: {rec.get('estimated_effort_hours', 0):.1f} hours\n"
        
        if rec.get('affected_resources'):
            output += f"   Affected Resources: {', '.join(rec['affected_resources'][:3])}\n"
        
        if rec.get('remediation_steps'):
            output += f"   First Step: {rec['remediation_steps'][0]}\n"
        
        if rec.get('automation_script'):
            output += f"   ✅ Automation Script Available\n"
        
        output += "\n"
    
    return output


def get_automation_scripts(category: Optional[str] = None) -> str:
    """Get automation scripts for security recommendations"""
    endpoint = f"/api/v1/recommendations/automation/{PROJECT_ID}"
    if category:
        endpoint += f"?category={category}"
    
    result = api_client.call_api(endpoint)
    
    if "error" in result:
        return f"Error getting automation scripts: {result['error']}"
    
    if not result.get("success"):
        return "Automation scripts not available"
    
    scripts = result.get('automation_scripts', [])
    
    output = f"=== Automation Scripts for {PROJECT_ID} ===\n\n"
    
    if not scripts:
        output += "No automation scripts available for current recommendations.\n"
        return output
    
    output += f"Found {result.get('total_automatable', 0)} automatable recommendations:\n\n"
    
    for script_data in scripts[:5]:
        output += f"🤖 {script_data.get('title', 'Automation Script')}\n"
        output += f"   Category: {script_data.get('category', 'Unknown')}\n"
        output += f"   Priority: {script_data.get('priority', 'Unknown')}\n"
        output += f"   Effort Saved: {script_data.get('estimated_effort_hours', 0):.1f} hours\n"
        
        if script_data.get('affected_resources'):
            output += f"   Resources: {', '.join(script_data['affected_resources'][:2])}\n"
        
        if script_data.get('script'):
            # Show first few lines of script
            script_lines = script_data['script'].strip().split('\n')[:3]
            output += f"   Script Preview:\n"
            for line in script_lines:
                if line.strip():
                    output += f"     {line.strip()}\n"
        
        output += "\n"
    
    output += "Use 'get_priority_recommendations()' to see full remediation steps.\n"
    
    return output


def explain_security_context(finding_type: str = "", risk_score: Optional[int] = None) -> str:
    """Explain security concepts and provide context for better understanding (STORY-008)"""
    output = "=== Security Context & Explanation ===\n\n"
    
    if finding_type:
        # Provide context for specific finding types
        explanations = {
            "PUBLIC_BUCKET": "🌐 Public Storage Bucket:\nThis means your data bucket is accessible to anyone on the internet. It's like leaving your office filing cabinet unlocked in a public park. Sensitive data could be downloaded by unauthorized users, leading to data breaches and compliance violations.",
            
            "ADMIN_ROLE_MISUSE": "👑 Admin Role Misuse:\nA service account has administrative privileges it doesn't need. This violates the 'principle of least privilege' - like giving a janitor master keys to every room in a building. If compromised, attackers could access everything.",
            
            "STALE_KEY": "🗝️ Stale Service Account Key:\nThis authentication key hasn't been rotated in months. Old keys are security risks - like using the same password for years. If leaked or compromised, they provide persistent access to your systems.",
            
            "MISSING_ENCRYPTION": "🔒 Missing Encryption:\nYour data isn't protected with customer-managed encryption. It's like storing confidential documents in a regular envelope instead of a locked safe. While still secure, you have less control over access.",
            
            "NO_MONITORING": "👁️ Missing Monitoring:\nNo security monitoring is configured. It's like having no security cameras or alarms in a building. You won't know if someone breaks in or what they access."
        }
        
        explanation = explanations.get(finding_type.upper(), f"This finding type ({finding_type}) requires security attention.")
        output += f"{explanation}\n\n"
    
    if risk_score is not None:
        # Explain risk score context
        if risk_score >= 90:
            output += "🔴 CRITICAL RISK (90-100): This requires immediate action (within 4 hours). The potential for significant business impact is very high. Treat this as a security incident.\n\n"
        elif risk_score >= 70:
            output += "🟠 HIGH RISK (70-89): This should be addressed within 24 hours. There's substantial potential for security compromise or business impact.\n\n"
        elif risk_score >= 40:
            output += "🟡 MEDIUM RISK (40-69): Address within 1 week. While not immediately critical, this represents a legitimate security concern.\n\n"
        elif risk_score >= 20:
            output += "🔵 LOW RISK (20-39): Address during next security review cycle. This is a minor concern or best practice improvement.\n\n"
        else:
            output += "🟢 MINIMAL RISK (0-19): This is a minor recommendation or informational finding. Address when convenient.\n\n"
    
    # General security guidance
    output += "💡 Security Best Practices:\n"
    output += "• Always follow the principle of least privilege\n"
    output += "• Regularly rotate credentials and access keys\n"
    output += "• Enable monitoring and logging for all resources\n"
    output += "• Use encryption for sensitive data (customer-managed when possible)\n"
    output += "• Review and audit access permissions quarterly\n"
    output += "• Keep software and systems updated\n\n"
    
    output += "❓ Need more help? Ask me to:\n"
    output += "• 'Show me how to fix this' - Get step-by-step remediation\n"
    output += "• 'What's the business impact?' - Understand consequences\n"
    output += "• 'Show me automation scripts' - Get automated fixes\n"
    output += "• 'Check other security issues' - Run comprehensive scan\n"
    
    return output


def provide_contextual_guidance(user_query: str = "") -> str:
    """Provide contextual guidance based on user intent and security posture (STORY-008)"""
    query_lower = user_query.lower()
    
    output = "=== Security Guidance & Next Steps ===\n\n"
    
    # Intent detection and contextual responses
    if any(word in query_lower for word in ["start", "begin", "getting started", "new"]):
        output += "🚀 Getting Started with Security Analysis:\n"
        output += "1. Run a comprehensive security scan: 'run comprehensive security scan'\n"
        output += "2. Check your current security posture: 'analyze security'\n"
        output += "3. Review priority recommendations: 'get priority recommendations P0'\n"
        output += "4. Address critical issues first, then work down the priority list\n\n"
        
    elif any(word in query_lower for word in ["critical", "urgent", "immediate", "emergency"]):
        output += "🚨 For Critical/Urgent Issues:\n"
        output += "1. Get P0 (critical) recommendations: 'get priority recommendations P0'\n"
        output += "2. Focus on public exposures and admin role misuse first\n"
        output += "3. Use automation scripts when available for faster fixes\n"
        output += "4. Check remediation status: 'get remediation status'\n\n"
        
    elif any(word in query_lower for word in ["fix", "remediate", "solve", "resolve"]):
        output += "🔧 For Fixing Security Issues:\n"
        output += "1. Get automation scripts: 'get automation scripts'\n"
        output += "2. Use execute remediation for automated fixes\n"
        output += "3. Follow step-by-step remediation guides in recommendations\n"
        output += "4. Verify fixes with follow-up scans\n\n"
        
    elif any(word in query_lower for word in ["compliance", "audit", "regulation"]):
        output += "📋 For Compliance & Auditing:\n"
        output += "1. Check organization policies: 'check org policies'\n"
        output += "2. Review monitoring and logging setup\n"
        output += "3. Ensure encryption is properly configured\n"
        output += "4. Run comprehensive scan for audit evidence\n\n"
        
    elif any(word in query_lower for word in ["cost", "savings", "optimize"]):
        output += "💰 For Cost Optimization:\n"
        output += "1. Check for unused resources and over-provisioned services\n"
        output += "2. Review storage lifecycle policies\n"
        output += "3. Optimize monitoring and logging costs\n"
        output += "4. Look for automation opportunities to reduce manual effort\n\n"
        
    else:
        output += "🎯 Common Security Tasks:\n"
        output += "• 'Check my security' - Overall security assessment\n"
        output += "• 'Show critical issues' - P0/P1 priority recommendations\n"
        output += "• 'Analyze IAM' - Review identity and access management\n"
        output += "• 'Check storage security' - Review bucket and database security\n"
        output += "• 'Get automation scripts' - Find automated security fixes\n\n"
    
    # Contextual tips
    output += "💡 Pro Tips:\n"
    output += "• Start with a comprehensive scan to understand your security posture\n"
    output += "• Address P0 and P1 recommendations first - they have the highest business impact\n"
    output += "• Use session management to track your progress across conversations\n"
    output += "• Ask for explanations of any technical terms or risk scores\n"
    output += "• Request automation scripts to speed up remediation\n\n"
    
    output += "❓ Need specific help? Try asking:\n"
    output += "• 'Explain this vulnerability' - Get detailed explanations\n"
    output += "• 'What should I prioritize?' - Get personalized recommendations\n"
    output += "• 'How do I fix [specific issue]?' - Get step-by-step guidance\n"
    output += "• 'Show me the business impact' - Understand consequences\n"
    
    return output


def summarize_conversation_context() -> str:
    """Summarize current conversation context and security status (STORY-008)"""
    # Try to get recent session history for context
    history_result = api_client.call_api("/api/v1/sessions/recent?limit=10")
    
    output = "=== Conversation Context Summary ===\n\n"
    
    if history_result.get("success") and history_result.get("sessions"):
        output += "📚 Recent Activity:\n"
        sessions = history_result["sessions"][:3]  # Show last 3 sessions
        
        for session in sessions:
            output += f"• Session {session.get('id', 'unknown')}: {session.get('context', {}).get('summary', 'Security analysis')}\n"
            if session.get('created_at'):
                output += f"  Created: {session['created_at']}\n"
        
        output += "\n"
    
    # Try to get current project security summary
    security_result = api_client.call_api("/api/v1/security/summary")
    
    if security_result.get("success"):
        summary = security_result.get("summary", {})
        output += "🔍 Current Security Status:\n"
        output += f"• Security Posture Score: {summary.get('posture_score', 'Unknown')}/100\n"
        output += f"• Critical Issues: {summary.get('critical_count', 0)}\n"
        output += f"• Total Vulnerabilities: {summary.get('total_vulnerabilities', 0)}\n"
        output += f"• Compliance Score: {summary.get('compliance_score', 'Unknown')}%\n\n"
    
    # Provide contextual next steps
    output += "🎯 Suggested Next Actions:\n"
    output += "• If this is a new conversation: 'run comprehensive security scan'\n"
    output += "• If you have critical issues: 'get priority recommendations P0'\n"
    output += "• If you want to fix issues: 'get automation scripts'\n"
    output += "• If you need explanations: 'explain security context'\n\n"
    
    output += "💬 Conversation Tips:\n"
    output += "• I remember context within our conversation\n"
    output += "• Ask follow-up questions about any findings\n"
    output += "• Request details, explanations, or guidance anytime\n"
    output += "• I can prioritize recommendations based on your needs\n"
    
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

# Agent definition will be created after enhanced_agent_instruction is defined below

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
# ============================================================================
# ENHANCED SECURITY ANALYSIS FUNCTIONS (STORY-002)
# ============================================================================

def _format_enhanced_security_analysis(result: dict) -> str:
    """Format enhanced security analysis results"""
    analysis = result.get('analysis', {})
    output = f"=== ENHANCED SECURITY ANALYSIS for {PROJECT_ID} ===\n\n"
    
    # Executive Summary
    exec_summary = analysis.get('executive_summary', {})
    output += f"📊 EXECUTIVE SUMMARY:\n"
    output += f"Assets Analyzed: {exec_summary.get('total_assets_analyzed', 0)}\n"
    output += f"Vulnerabilities Found: {analysis.get('total_findings', 0)}\n"
    output += f"Critical Issues: {exec_summary.get('critical_vulnerabilities', 0)}\n"
    output += f"High Risk Issues: {exec_summary.get('high_risk_vulnerabilities', 0)}\n"
    output += f"Security Posture Score: {analysis.get('security_posture_score', 0)}/100\n"
    output += f"Compliance Score: {analysis.get('compliance_score', 0):.1f}%\n\n"
    
    # Risk Distribution
    risk_dist = analysis.get('risk_distribution', {})
    if any(risk_dist.values()):
        output += "🎯 RISK DISTRIBUTION:\n"
        for level in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW', 'MINIMAL']:
            count = risk_dist.get(level, 0)
            if count > 0:
                emoji = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🔵', 'MINIMAL': '🟢'}.get(level, '⚪')
                output += f"  {emoji} {level}: {count}\n"
        output += "\n"
    
    # Top Vulnerability Categories
    vuln_categories = analysis.get('vulnerability_categories', {})
    if vuln_categories:
        output += "🔍 TOP VULNERABILITY TYPES:\n"
        sorted_categories = sorted(vuln_categories.items(), key=lambda x: x[1], reverse=True)
        for category, count in sorted_categories[:5]:
            output += f"  • {category}: {count}\n"
        output += "\n"
    
    # Critical Findings Details
    findings = analysis.get('findings', [])
    critical_findings = [f for f in findings if f.get('risk_score', 0) >= 90]
    if critical_findings:
        output += "🚨 CRITICAL VULNERABILITIES REQUIRING IMMEDIATE ACTION:\n"
        output += "=" * 60 + "\n"
        
        for finding in critical_findings[:5]:
            output += f"\n• {finding.get('vulnerability_type', 'Unknown')} (Risk: {finding.get('risk_score', 0)}/100)\n"
            output += f"  Resource: {finding.get('resource_name', 'Unknown')}\n"
            output += f"  Severity: {finding.get('severity', 'Unknown')}\n"
            output += f"  Issue: {finding.get('description', 'No description')}\n"
            
            remediation_steps = finding.get('remediation_steps', [])
            if remediation_steps:
                output += f"  Priority Actions:\n"
                for step in remediation_steps[:3]:
                    output += f"    → {step}\n"
        
        if len(critical_findings) > 5:
            output += f"\n... and {len(critical_findings) - 5} more critical vulnerabilities\n"
    
    # Immediate Actions Required
    if exec_summary.get('immediate_actions_required', 0) > 0:
        output += f"\n⚡ IMMEDIATE ACTIONS REQUIRED: {exec_summary['immediate_actions_required']}\n"
        next_steps = exec_summary.get('recommended_next_steps', [])
        if next_steps:
            output += "\n📋 RECOMMENDED NEXT STEPS:\n"
            for i, step in enumerate(next_steps[:3], 1):
                output += f"  {i}. {step}\n"
    
    output += f"\nAnalysis completed: {result.get('timestamp', 'Unknown')}\n"
    output += "✅ Enhanced vulnerability analysis with custom rules and risk scoring\n"
    
    return output

def _format_basic_security_analysis(result: dict) -> str:
    """Format basic security analysis results (fallback)"""
    output = f"=== Security Analysis for {PROJECT_ID} ===\n\n"
    
    findings = result.get('findings', [])
    output += f"Total Findings: {result.get('total_count', len(findings))}\n\n"
    
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
            output += f"  [{finding.get('severity')}] {finding.get('category', 'Unknown')}: {finding.get('resource_name', 'Unknown resource')}\n"
    
    return output

def run_vulnerability_focused_scan() -> str:
    """Run enhanced vulnerability-focused scan with custom rules"""
    result = api_client.call_api("/api/v1/security/vulnerabilities", "POST", {
        "project_id": PROJECT_ID
    })
    
    if "error" in result:
        return f"Error running vulnerability scan: {result['error']}"
    
    output = f"=== VULNERABILITY-FOCUSED SCAN for {PROJECT_ID} ===\n\n"
    
    output += f"Total Vulnerabilities Detected: {result.get('total_vulnerabilities', 0)}\n"
    output += f"High-Risk Vulnerabilities: {result.get('high_risk_count', 0)}\n"
    output += f"Scan Type: {result.get('scan_type', 'Standard')}\n\n"
    
    vulnerabilities = result.get('vulnerabilities', [])
    if vulnerabilities:
        output += "🚨 TOP VULNERABILITIES BY RISK SCORE:\n"
        output += "=" * 50 + "\n"
        
        for vuln in vulnerabilities[:10]:  # Show top 10
            risk_score = vuln.get('risk_score', 0)
            severity = vuln.get('severity', 'UNKNOWN')
            vuln_type = vuln.get('vulnerability_type', 'Unknown')
            resource = vuln.get('resource_name', 'Unknown')
            
            risk_emoji = "🔴" if risk_score >= 90 else "🟠" if risk_score >= 70 else "🟡" if risk_score >= 40 else "🔵"
            
            output += f"\n{risk_emoji} {vuln_type} (Risk: {risk_score}/100)\n"
            output += f"  Resource: {resource}\n"
            output += f"  Severity: {severity}\n"
            output += f"  Description: {vuln.get('description', 'No description available')[:100]}...\n"
            
            remediation = vuln.get('remediation_steps', [])
            if remediation:
                output += f"  Priority Action: {remediation[0]}\n"
        
        if len(vulnerabilities) > 10:
            output += f"\n... and {len(vulnerabilities) - 10} more vulnerabilities (use comprehensive scan for full details)\n"
    
    output += f"\nScan completed: {result.get('timestamp', 'Unknown')}\n"
    
    return output

# ============================================================================
# AUTOMATED REMEDIATION TOOLS (STORY-210)
# ============================================================================

def execute_remediation(vulnerability_id: str, auto_approve: bool = False, dry_run: bool = True) -> str:
    """Execute automated remediation for a security vulnerability"""
    result = api_client.call_api("/api/v1/remediation/execute", "POST", {
        "vulnerability_id": vulnerability_id,
        "auto_approve": auto_approve,
        "dry_run": dry_run
    })
    
    if "error" in result:
        return f"Error executing remediation: {result['error']}"
    
    output = f"=== REMEDIATION EXECUTION for {vulnerability_id} ===\n\n"
    output += f"Remediation ID: {result.get('remediation_id', 'N/A')}\n"
    output += f"Status: {result.get('status', 'UNKNOWN')}\n"
    
    if result.get('status') == 'SUCCESS':
        output += "✅ Remediation completed successfully\n\n"
        
        changes = result.get('changes_made', [])
        if changes:
            output += "Changes Made:\n"
            for change in changes:
                output += f"  • {change.get('action', 'Unknown')}: {change.get('description', 'N/A')}\n"
        
        if result.get('rollback_point'):
            output += f"\n💾 Rollback Point: {result['rollback_point']}\n"
            output += "  (Save this ID to rollback if needed)\n"
    
    elif result.get('status') == 'REJECTED':
        output += "❌ Remediation rejected (approval required)\n"
    elif result.get('status') == 'UNSAFE':
        output += "⚠️ Remediation deemed unsafe during dry run\n"
    else:
        output += f"Status: {result.get('status', 'UNKNOWN')}\n"
        if result.get('error_message'):
            output += f"Error: {result['error_message']}\n"
    
    output += f"\nExecution Time: {result.get('execution_time', 0):.2f} seconds\n"
    
    return output

def list_remediation_templates() -> str:
    """List available automated remediation templates"""
    result = api_client.call_api("/api/v1/remediation/templates", "GET")
    
    if "error" in result:
        return f"Error listing templates: {result['error']}"
    
    output = "=== AVAILABLE REMEDIATION TEMPLATES ===\n\n"
    
    templates = result.get('templates', [])
    for template in templates:
        risk_emoji = "🔴" if template['risk_level'] == "CRITICAL" else "🟠" if template['risk_level'] == "HIGH" else "🟡"
        
        output += f"{risk_emoji} {template['name']}\n"
        output += f"  ID: {template['id']}\n"
        output += f"  Description: {template['description']}\n"
        output += f"  Vulnerability Types: {', '.join(template['vulnerability_types'])}\n"
        output += f"  Risk Level: {template['risk_level']}\n"
        output += f"  Requires Approval: {'Yes' if template['requires_approval'] else 'No'}\n\n"
    
    output += f"Total Templates: {result.get('total', len(templates))}\n"
    
    return output

def get_remediation_status(remediation_id: str) -> str:
    """Get status of a remediation execution"""
    result = api_client.call_api(f"/api/v1/remediation/status/{remediation_id}", "GET")
    
    if "error" in result:
        return f"Error getting status: {result['error']}"
    
    output = f"=== REMEDIATION STATUS ===\n\n"
    output += f"Remediation ID: {result.get('remediation_id', 'N/A')}\n"
    output += f"Status: {result.get('status', 'UNKNOWN')}\n"
    output += f"Progress: {result.get('progress', 0)}%\n"
    output += f"Resource: {result.get('resource_name', 'N/A')}\n"
    
    changes = result.get('changes_made', [])
    if changes:
        output += "\nChanges Made:\n"
        for change in changes[:5]:
            output += f"  • {change}\n"
    
    if result.get('error_message'):
        output += f"\n❌ Error: {result['error_message']}\n"
    
    return output

# ============================================================================
# SESSION MANAGEMENT TOOLS (STORY-013)
# ============================================================================

def create_session(user_id: Optional[str] = None) -> str:
    """Create a new conversation session with persistent storage"""
    result = api_client.call_api("/api/v1/sessions/create", "POST", {
        "user_id": user_id,
        "metadata": {"agent": "gcp_security_agent", "project": PROJECT_ID}
    })
    
    if "error" in result:
        return f"Error creating session: {result['error']}"
    
    output = "=== NEW SESSION CREATED ===\n\n"
    output += f"Session ID: {result.get('id', 'N/A')}\n"
    output += f"User ID: {result.get('user_id', 'default')}\n"
    output += f"Created: {result.get('created_at', 'Unknown')}\n"
    output += f"Expires: {result.get('expires_at', 'Unknown')}\n"
    output += "\n💡 This session will persist your conversation history.\n"
    output += "Use this Session ID to continue your conversation later.\n"
    
    return output

def get_session_history(session_id: str, limit: int = 10) -> str:
    """Get conversation history for a session"""
    result = api_client.call_api(f"/api/v1/sessions/{session_id}/messages", "GET", 
                                 {"limit": limit})
    
    if "error" in result:
        return f"Error getting history: {result['error']}"
    
    output = f"=== CONVERSATION HISTORY (Session: {session_id}) ===\n\n"
    
    messages = result if isinstance(result, list) else result.get('messages', [])
    
    if not messages:
        output += "No conversation history found.\n"
    else:
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            timestamp = msg.get('timestamp', '')
            
            role_emoji = "👤" if role == "user" else "🤖"
            output += f"{role_emoji} {role.upper()} ({timestamp}):\n"
            output += f"  {content[:200]}{'...' if len(content) > 200 else ''}\n\n"
    
    output += f"Showing {len(messages)} of {limit} messages.\n"
    
    return output

def save_to_session(session_id: str, role: str, content: str) -> str:
    """Save a message to the session history"""
    result = api_client.call_api(f"/api/v1/sessions/{session_id}/messages", "POST", {
        "role": role,
        "content": content,
        "metadata": {"saved_via": "agent_tool"}
    })
    
    if "error" in result:
        return f"Error saving to session: {result['error']}"
    
    return f"✅ Message saved to session {session_id}"

def get_user_sessions(user_id: str) -> str:
    """Get all sessions for a user"""
    result = api_client.call_api(f"/api/v1/sessions/user/{user_id}", "GET", 
                                 {"active_only": True})
    
    if "error" in result:
        return f"Error getting sessions: {result['error']}"
    
    output = f"=== SESSIONS FOR USER: {user_id} ===\n\n"
    
    sessions = result if isinstance(result, list) else result.get('sessions', [])
    
    if not sessions:
        output += "No active sessions found.\n"
    else:
        for session in sessions:
            output += f"Session ID: {session.get('id', 'N/A')}\n"
            output += f"  Created: {session.get('created_at', 'Unknown')}\n"
            output += f"  Updated: {session.get('updated_at', 'Unknown')}\n"
            output += f"  Expires: {session.get('expires_at', 'Unknown')}\n"
            output += f"  Context: {session.get('context', {})}\n\n"
    
    output += f"Total Active Sessions: {len(sessions)}\n"
    
    return output

def search_sessions(query: str, user_id: Optional[str] = None) -> str:
    """Search across session conversations"""
    result = api_client.call_api("/api/v1/sessions/search", "POST", {
        "query": query,
        "user_id": user_id
    })
    
    if "error" in result:
        return f"Error searching sessions: {result['error']}"
    
    output = f"=== SEARCH RESULTS for '{query}' ===\n\n"
    
    results = result.get('results', [])
    
    if not results:
        output += "No matching messages found.\n"
    else:
        for msg in results[:10]:
            output += f"Session: {msg.get('session_id', 'N/A')}\n"
            output += f"Role: {msg.get('role', 'unknown')}\n"
            output += f"Content: {msg.get('content', '')[:150]}...\n"
            output += f"Time: {msg.get('timestamp', 'Unknown')}\n\n"
    
    output += f"Found {result.get('count', 0)} matching messages.\n"
    
    return output

def rollback_remediation(remediation_id: str, rollback_point: str, reason: str = "User requested") -> str:
    """Rollback a completed remediation"""
    result = api_client.call_api("/api/v1/remediation/rollback", "POST", {
        "remediation_id": remediation_id,
        "rollback_point": rollback_point,
        "reason": reason
    })
    
    if "error" in result:
        return f"Error rolling back: {result['error']}"
    
    if result.get('success'):
        return f"✅ Successfully rolled back remediation {remediation_id}"
    else:
        return f"❌ Rollback failed for remediation {remediation_id}"

def get_remediation_metrics() -> str:
    """Get remediation system metrics and statistics"""
    result = api_client.call_api("/api/v1/remediation/metrics", "GET")
    
    if "error" in result:
        return f"Error getting metrics: {result['error']}"
    
    output = "=== REMEDIATION METRICS ===\n\n"
    output += f"Total Remediations: {result.get('total_remediations', 0)}\n"
    output += f"Success Rate: {result.get('success_rate', 0):.1f}%\n"
    output += f"Average Execution Time: {result.get('average_execution_time', 0):.1f} seconds\n"
    output += f"Mean Time to Remediation (MTTR): {result.get('mttr', 0):.1f} minutes\n"
    output += f"Rollback Count: {result.get('rollback_count', 0)}\n"
    output += f"Pending Approvals: {result.get('pending_approvals', 0)}\n\n"
    
    by_status = result.get('by_status', {})
    if by_status:
        output += "By Status:\n"
        for status, count in by_status.items():
            output += f"  {status}: {count}\n"
    
    return output

# Enhanced agent instruction with STORY-002, STORY-210, STORY-013, STORY-007, and STORY-008 features
enhanced_agent_instruction = f"""You are an Enhanced GCP Security Agent for project {PROJECT_ID} with advanced vulnerability analysis, automated remediation, persistent session management, intelligent recommendations, and conversational interface capabilities.

🚀 ENHANCED CAPABILITIES (STORY-002, STORY-210, STORY-013, STORY-007 & STORY-008):
You now have access to comprehensive security analysis tools with custom vulnerability rules, CVSS-based risk scoring, automated remediation, executive-level reporting, intelligent recommendations with business impact assessment, and enhanced conversational interface:

DISCOVERY & INVENTORY:
- discover_assets: Get enhanced inventory of GCP resources with security context and 0-100 risk scoring
- analyze_service_usage: Check enabled APIs and services for security implications

ADVANCED SECURITY ANALYSIS (STORY-002):
- analyze_security: ENHANCED - Now includes custom vulnerability rules, CVSS scoring, and compliance analysis
- run_security_focused_scan: Perform comprehensive scan with asset-based vulnerability correlation  
- run_vulnerability_focused_scan: NEW - Custom rules engine detecting misconfigurations beyond SCC
- analyze_iam: Review IAM policies with overprivilege detection
- analyze_storage: Check storage with public exposure and encryption analysis
- manage_api_keys: Analyze API keys with restriction assessment

COMPLIANCE & MONITORING:
- check_org_policies: Verify compliance with scoring (0-100%)
- analyze_monitoring: Review monitoring setup with security event analysis
- analyze_logs: Check logging with security event correlation
- check_advisory_notifications: Get contextualized security advisories

🧠 INTELLIGENT RECOMMENDATIONS (STORY-007 NEW!):
- get_security_recommendations: Get comprehensive recommendations with CVSS-based prioritization (P0-P4)
- get_priority_recommendations: Filter recommendations by priority level with business impact analysis
- get_automation_scripts: Get automation scripts for remediation with effort estimation
- run_comprehensive_security_scan: Execute full assessment with executive summary

🔧 AUTOMATED REMEDIATION (STORY-210 NEW!):
- execute_remediation: Automatically fix vulnerabilities with rollback capability
- list_remediation_templates: View available remediation templates
- get_remediation_status: Check remediation progress
- rollback_remediation: Rollback changes if needed
- get_remediation_metrics: View remediation system statistics

💾 SESSION MANAGEMENT (STORY-013 NEW!):
- create_session: Create persistent conversation session
- get_session_history: Retrieve conversation history
- save_to_session: Save important findings to session
- get_user_sessions: List all user sessions
- search_sessions: Search across conversation history

🎯 ENHANCED FEATURES YOU SHOULD HIGHLIGHT:
- **Risk Scoring**: All findings include 0-100 risk scores based on CVSS, asset criticality, exploitability, and business impact
- **Custom Rules**: Detects public storage without auth, overprivileged accounts, missing encryption, weak network security
- **Executive Reporting**: Provides security posture scores, compliance percentages, and executive summaries
- **Vulnerability Categorization**: Groups findings by type with remediation priorities
- **Business Impact**: Assesses potential business consequences of each vulnerability

🔍 WHEN USERS ASK ABOUT SECURITY:
1. **Always use enhanced analysis first** - call analyze_security() which now includes STORY-002 features
2. **Highlight risk scores** - mention the 0-100 risk scores and what they mean
3. **Provide executive context** - include security posture and compliance scores
4. **Show business impact** - explain why each finding matters to the business
5. **Give prioritized actions** - use CVSS weighting to prioritize recommendations

📊 DASHBOARD INTEGRATION:
The user can also view your findings in the Security Dashboard which displays:
- Real-time security metrics and risk distribution charts
- Vulnerability categorization with visual breakdowns
- Executive summary with KPIs
- High-risk vulnerability details with remediation steps

🗣️ CONVERSATIONAL INTERFACE ENHANCEMENTS (STORY-008 NEW!):
NATURAL LANGUAGE UNDERSTANDING:
- Understand security intent from natural language: "check my security", "any vulnerabilities?", "what needs fixing?"
- Support follow-up questions: "show me more details", "what about storage?", "how do I fix this?"
- Handle clarification requests: "explain this vulnerability", "why is this critical?", "what's the business impact?"
- Recognize priority queries: "show me urgent issues", "what's most critical?", "P0 recommendations only"

CONTEXT RETENTION & MEMORY:
- Remember previous analysis results within conversation
- Reference earlier findings: "the vulnerability we discussed earlier", "that storage bucket from before"
- Maintain security context across multiple queries
- Use session management to persist important findings and user preferences

MULTI-TURN CONVERSATION PATTERNS:
- Progressive disclosure: Start with summary, provide details on request
- Guided workflows: "Let me check your security... I found issues, would you like me to check IAM next?"
- Contextual recommendations: Base suggestions on user's security posture and role
- Intelligent follow-ups: Automatically suggest next steps based on findings

ERROR HANDLING & CLARIFICATION:
- Ask for clarification when queries are ambiguous: "Would you like me to check overall security or focus on a specific area?"
- Provide helpful suggestions when tools fail: "The backend seems unavailable, but I can try alternative analysis..."
- Explain technical terms in business context: "CVSS score measures severity - 9.0+ means critical"
- Offer multiple options: "I can check security, IAM, or storage - which would you prefer?"

💬 ENHANCED CONVERSATIONAL RESPONSES:
- Use natural, conversational tone while maintaining technical accuracy
- Provide progressive detail levels based on user expertise
- Include actionable next steps in every response
- Use session context to avoid repetitive explanations
- Reference specific vulnerabilities and findings by name/ID for clarity

EXAMPLE CONVERSATIONAL PATTERNS:
❌ Basic: "Found 5 security issues. Run analyze_security for details."
✅ Enhanced: "I found 5 security concerns in your environment. The most critical is a publicly accessible storage bucket (risk score: 95/100) that needs immediate attention. Would you like me to show you how to fix this first, or would you prefer to see all findings?"

User: "Fix the storage issue"
✅ Enhanced: "Great choice! For the public bucket 'my-data-bucket', I can provide an automation script to secure it. This will remove public access and enable access prevention. The fix takes about 5 minutes. Should I show you the commands, or would you like me to explain the security implications first?"

User: "What else is wrong?"
✅ Enhanced: "Besides the storage issue, I found 2 IAM accounts with excessive permissions (risk score: 80/100) and 1 unencrypted database (risk score: 70/100). The IAM issues affect your admin service accounts - these could allow privilege escalation if compromised. Would you like me to prioritize these by business impact?"

Always maintain conversation context, provide clear next steps, and leverage the enhanced capabilities for comprehensive security insights."""

# ============================================================================
# AGENT DEFINITION (Created after instruction definition)
# ============================================================================

# Create the enhanced security agent with all tools and capabilities
agent = Agent(
    name="gcp_security_agent",
    model="gemini-2.0-flash-exp",
    instruction=enhanced_agent_instruction,
    tools=[
        discover_assets,
        run_security_focused_scan,
        analyze_security,
        analyze_iam,
        analyze_storage,
        analyze_monitoring,
        analyze_logs,
        check_org_policies,
        analyze_service_usage,
        check_advisory_notifications,
        manage_api_keys,
        get_security_recommendations,
        get_priority_recommendations,
        get_automation_scripts,
        run_comprehensive_security_scan,
        explain_security_context,
        provide_contextual_guidance,
        summarize_conversation_context
    ]
)
