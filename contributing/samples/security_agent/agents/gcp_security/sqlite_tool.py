"""
SQLite Query Tool for Vertex AI Agent
=====================================

Single tool that can query all cached GCP security data from SQLite.
This works around Vertex AI's single-tool limitation by providing
comprehensive data access through SQL queries.
"""

import sqlite3
import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)

# Database configuration
# Get the database path relative to the security_agent directory
import os
from pathlib import Path

# Find the security_agent root directory
current_file = Path(__file__)
security_agent_dir = current_file.parent.parent.parent  # Up to security_agent/
DB_PATH = os.getenv('DATABASE_PATH', str(security_agent_dir / 'backend' / 'cache' / 'gcp_data.db'))

def query_security_data(query_type: str, parameters: Optional[str] = None) -> str:
    """
    Query GCP security data from SQLite cache.
    
    This is the single tool that provides access to all cached security data.
    The agent can request different types of data by specifying the query_type.
    
    Args:
        query_type: Type of query to execute. Options include:
            - 'assets': List all GCP assets
            - 'security_findings': Get security findings
            - 'iam_analysis': Analyze IAM permissions
            - 'storage_buckets': List and analyze storage buckets
            - 'api_keys': List API keys
            - 'recommendations': Get security recommendations
            - 'org_policies': Check organization policies
            - 'service_usage': Analyze service usage
            - 'monitoring': Get monitoring data
            - 'logs': Get audit logs
            - 'cache_status': Show cache statistics
            - 'msa_analysis': View MSA (Monthly Service Announcement) analysis history
            - 'msa_changes': Query specific MSA changes and their details
            - 'context_aware_analysis': Full feedback loop analysis connecting MSA changes with security findings, assets, and remediation effectiveness
            - 'cross_impact_analysis': Analyze how changes in one area affect other security domains
            - 'msa_impact': Get MSA impact assessments for projects
            - 'knowledge_base': Query enterprise policies, coding standards, and best practices
            - 'coding_standards': Query coding standards and test requirements
            - 'enterprise_policies': Query security and governance policies
            - 'best_practices': Query GCP best practices and recommendations
            - 'compliance': Query compliance framework requirements
            - 'custom': Execute a custom SQL query (be careful!)
            
        parameters: Optional JSON string with query parameters.
            Examples:
            - For 'assets': '{"asset_type": "compute.googleapis.com/Instance"}'
            - For 'security_findings': '{"severity": "HIGH"}'
            - For 'iam_analysis': '{"principal": "user@example.com"}'
            - For 'storage_buckets': '{"bucket_name": "my-bucket"}'
            - For 'custom': '{"sql": "SELECT * FROM assets LIMIT 10"}'
    
    Returns:
        String with query results formatted as readable text
    
    Examples:
        >>> query_security_data("assets")
        "Found 25 assets in project..."
        
        >>> query_security_data("security_findings", '{"severity": "HIGH"}')
        "Found 3 HIGH severity findings..."
    """
    
    # Parse parameters
    params = {}
    if parameters:
        try:
            params = json.loads(parameters) if isinstance(parameters, str) else parameters
        except json.JSONDecodeError:
            params = {'value': parameters}
    
    # Ensure database exists
    db_path = Path(DB_PATH)
    if not db_path.exists():
        db_path.parent.mkdir(parents=True, exist_ok=True)
        return "❌ Database not found. Please run data refresh first to populate the cache."
    
    try:
        conn = sqlite3.connect(str(db_path))
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Route to appropriate query based on type
        if query_type == 'assets':
            return _query_assets(cursor, params)
        elif query_type == 'security_findings':
            return _query_security_findings(cursor, params)
        elif query_type == 'iam_analysis':
            return _query_iam(cursor, params)
        elif query_type == 'storage_buckets':
            return _query_storage(cursor, params)
        elif query_type == 'api_keys':
            return _query_api_keys(cursor, params)
        elif query_type == 'recommendations':
            return _query_recommendations(cursor, params)
        elif query_type == 'org_policies':
            return _query_org_policies(cursor, params)
        elif query_type == 'service_usage':
            return _query_service_usage(cursor, params)
        elif query_type == 'monitoring':
            return _query_monitoring(cursor, params)
        elif query_type == 'logs':
            return _query_logs(cursor, params)
        elif query_type == 'cache_status':
            return _get_cache_status(cursor)
        elif query_type == 'security_summary':
            return _get_security_summary(cursor, params)
        elif query_type == 'firewall_rules':
            return _query_firewall_rules(cursor, params)
        elif query_type == 'networks':
            return _query_networks(cursor, params)
        elif query_type == 'compute_instances':
            return _query_compute_instances(cursor, params)
        elif query_type == 'databases':
            return _query_databases(cursor, params)
        elif query_type == 'iam_accounts':
            return _query_iam_accounts(cursor, params)
        elif query_type == 'secrets':
            return _query_secrets(cursor, params)
        elif query_type == 'msa_analysis':
            return _query_msa_analysis(cursor, params)
        elif query_type == 'msa_changes':
            return _query_msa_changes(cursor, params)
        elif query_type == 'context_aware_analysis':
            return _query_context_aware_analysis(cursor, params)
        elif query_type == 'cross_impact_analysis':
            return _query_cross_impact_analysis(cursor, params)
        elif query_type == 'msa_impact':
            return _query_msa_impact(cursor, params)
        elif query_type == 'msa_permissions':
            return _query_msa_permissions(cursor, params)
        elif query_type == 'knowledge_base':
            return _query_knowledge_base(cursor, params)
        elif query_type == 'coding_standards':
            return _query_coding_standards(cursor, params)
        elif query_type == 'enterprise_policies':
            return _query_enterprise_policies(cursor, params)
        elif query_type == 'best_practices':
            return _query_best_practices(cursor, params)
        elif query_type == 'compliance':
            return _query_compliance(cursor, params)
        elif query_type == 'custom':
            return _execute_custom_query(cursor, params)
        else:
            return f"❌ Unknown query type: {query_type}\n\nAvailable types: security_summary, assets, security_findings, iam_analysis, storage_buckets, api_keys, recommendations, org_policies, service_usage, monitoring, logs, firewall_rules, networks, compute_instances, databases, iam_accounts, secrets, msa_analysis, msa_changes, msa_impact, knowledge_base, coding_standards, enterprise_policies, best_practices, compliance, cache_status, custom"
            
    except Exception as e:
        logger.error(f"Database query error: {str(e)}")
        return f"❌ Database error: {str(e)}"
    finally:
        if 'conn' in locals():
            conn.close()

def _query_assets(cursor, params: Dict) -> str:
    """Query GCP assets"""
    asset_type = params.get('asset_type', '')
    
    if asset_type:
        cursor.execute("""
            SELECT name, asset_type, location, labels, create_time
            FROM assets 
            WHERE asset_type = ?
            ORDER BY create_time DESC
        """, (asset_type,))
    else:
        cursor.execute("""
            SELECT asset_type, COUNT(*) as count
            FROM assets
            GROUP BY asset_type
            ORDER BY count DESC
        """)
    
    results = cursor.fetchall()
    
    if not results:
        return "No assets found in cache."
    
    if asset_type:
        output = f"📦 Assets of type {asset_type}:\n\n"
        for row in results:
            output += f"• {row['name']}\n"
            output += f"  Location: {row['location']}\n"
            output += f"  Created: {row['create_time']}\n"
            if row['labels']:
                output += f"  Labels: {row['labels']}\n"
            output += "\n"
    else:
        output = "📊 Asset Summary:\n\n"
        total = sum(row['count'] for row in results)
        output += f"Total assets: {total}\n\n"
        for row in results:
            output += f"• {row['asset_type']}: {row['count']}\n"
    
    return output

def _query_security_findings(cursor, params: Dict) -> str:
    """Query security findings"""
    severity = params.get('severity', '')
    category = params.get('category', '')
    
    query = "SELECT * FROM security_findings WHERE 1=1"
    query_params = []
    
    if severity:
        query += " AND severity = ?"
        query_params.append(severity)
    if category:
        query += " AND category = ?"
        query_params.append(category)
    
    query += """ ORDER BY CASE severity 
                       WHEN 'CRITICAL' THEN 1 
                       WHEN 'HIGH' THEN 2 
                       WHEN 'MEDIUM' THEN 3 
                       WHEN 'LOW' THEN 4 
                       ELSE 5 END, 
                   event_time DESC LIMIT 20"""
    
    cursor.execute(query, query_params)
    results = cursor.fetchall()
    
    if not results:
        return "No security findings found."
    
    output = f"🔍 Security Findings ({len(results)} found):\n\n"
    
    # Group by severity
    by_severity = {}
    for row in results:
        sev = row['severity']
        if sev not in by_severity:
            by_severity[sev] = []
        by_severity[sev].append(row)
    
    severity_order = ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']
    for sev in severity_order:
        if sev in by_severity:
            output += f"\n🔴 {sev} Severity ({len(by_severity[sev])}):\n"
            for finding in by_severity[sev][:5]:  # Show top 5 per severity
                output += f"  • {finding['category']}: {finding['resource_name']}\n"
                output += f"    {finding['description'][:100]}...\n"
    
    return output

def _query_iam(cursor, params: Dict) -> str:
    """Query IAM data"""
    principal = params.get('principal', '')
    
    if principal:
        cursor.execute("""
            SELECT resource_type, resource_name, role, condition
            FROM iam_policies
            WHERE member LIKE ?
            ORDER BY resource_type, resource_name
        """, (f'%{principal}%',))
    else:
        # First get risky roles
        cursor.execute("""
            SELECT member, role, COUNT(*) as grant_count
            FROM iam_policies
            WHERE role IN ('roles/owner', 'roles/editor', 'roles/iam.securityAdmin', 
                          'roles/resourcemanager.organizationAdmin', 'roles/compute.admin',
                          'roles/storage.admin', 'roles/iam.serviceAccountAdmin',
                          'roles/iam.serviceAccountKeyAdmin', 'roles/iam.roleAdmin')
            GROUP BY member, role
            ORDER BY grant_count DESC
            LIMIT 10
        """)
        risky_results = cursor.fetchall()
        
        # Then get role summary
        cursor.execute("""
            SELECT role, COUNT(DISTINCT resource_name) as resource_count,
                   COUNT(DISTINCT member) as member_count
            FROM iam_policies
            GROUP BY role
            ORDER BY member_count DESC
            LIMIT 15
        """)
    
    results = cursor.fetchall()
    
    if not results and not principal:
        return "No IAM data found."
    
    if principal:
        if not results:
            return f"No IAM permissions found for {principal}."
        output = f"👤 IAM Analysis for {principal}:\n\n"
        for row in results:
            output += f"• Resource: {row['resource_type']}/{row['resource_name']}\n"
            output += f"  Role: {row['role']}\n"
            if row['condition']:
                output += f"  Condition: {row['condition']}\n"
            output += "\n"
    else:
        output = "🔐 IAM Analysis:\n\n"
        
        if risky_results:
            output += "⚠️ High-Risk Permissions:\n"
            for row in risky_results:
                member_type = "Service Account" if "@" in row['member'] and ".iam.gserviceaccount.com" in row['member'] else "User/Group"
                output += f"  • {row['member'][:50]}... ({member_type})\n"
                output += f"    Role: {row['role']} ({row['grant_count']} grants)\n"
            output += "\n"
        
        output += "📊 Role Distribution:\n"
        for row in results:
            output += f"  • {row['role']}:\n"
            output += f"    {row['member_count']} members on {row['resource_count']} resources\n"
    
    return output

def _query_storage(cursor, params: Dict) -> str:
    """Query storage bucket data"""
    bucket_name = params.get('bucket_name', '')
    
    if bucket_name:
        cursor.execute("""
            SELECT * FROM storage_buckets
            WHERE name = ?
        """, (bucket_name,))
        result = cursor.fetchone()
        
        if not result:
            return f"Bucket {bucket_name} not found."
        
        output = f"🪣 Storage Bucket: {bucket_name}\n\n"
        output += f"Location: {result['location']}\n"
        output += f"Storage Class: {result['storage_class']}\n"
        output += f"Versioning: {'Enabled' if result['versioning_enabled'] else 'Disabled'}\n"
        output += f"Public Access: {result['public_access']}\n"
        output += f"Uniform Bucket Level Access: {'Enabled' if result['uniform_bucket_level_access'] else 'Disabled'}\n"
        output += f"Encryption: {result['encryption']}\n"
        if result['labels']:
            output += f"Labels: {result['labels']}\n"
        output += f"Fetched: {result['fetched_at']}\n"
        
    else:
        cursor.execute("""
            SELECT 
                COUNT(*) as total,
                SUM(CASE WHEN versioning_enabled = 0 THEN 1 ELSE 0 END) as no_versioning,
                SUM(CASE WHEN uniform_bucket_level_access = 0 THEN 1 ELSE 0 END) as no_uniform_access
            FROM storage_buckets
        """)
        result = cursor.fetchone()
        
        output = "🪣 Storage Buckets Summary:\n\n"
        output += f"Total buckets: {result['total']}\n\n"
        output += "⚠️ Potential Issues:\n"
        output += f"  • Buckets without versioning: {result['no_versioning']}\n"
        output += f"  • Buckets without uniform bucket-level access: {result['no_uniform_access']}\n"
        
        # Show individual buckets
        cursor.execute("""
            SELECT name, location, storage_class, public_access, versioning_enabled
            FROM storage_buckets
            ORDER BY name
        """)
        buckets = cursor.fetchall()
        
        if buckets:
            output += f"\n📋 All Buckets ({len(buckets)}):\n"
            for bucket in buckets:
                security_status = "🔒 Private" if bucket['public_access'] != 'public' else "⚠️ Public"
                versioning_status = "✅" if bucket['versioning_enabled'] else "❌"
                output += f"  • {bucket['name']} ({bucket['location']}) - {security_status}, Versioning: {versioning_status}\n"
    
    return output

def _query_api_keys(cursor, params: Dict) -> str:
    """Query API keys"""
    cursor.execute("""
        SELECT name, display_name, create_time, update_time, restrictions
        FROM api_keys
        ORDER BY create_time DESC
    """)
    results = cursor.fetchall()
    
    if not results:
        return "No API keys found."
    
    output = f"🔑 API Keys ({len(results)} found):\n\n"
    for row in results:
        output += f"• {row['display_name'] or row['name']}\n"
        output += f"  Created: {row['create_time']}\n"
        if row['restrictions']:
            output += f"  Restrictions: {row['restrictions']}\n"
        output += "\n"
    
    return output

def _query_recommendations(cursor, params: Dict) -> str:
    """Query security recommendations"""
    cursor.execute("""
        SELECT recommender, priority, description, state
        FROM recommendations
        WHERE state = 'ACTIVE'
        ORDER BY 
            CASE priority 
                WHEN 'P1' THEN 1 
                WHEN 'P2' THEN 2 
                WHEN 'P3' THEN 3 
                ELSE 4 
            END
        LIMIT 10
    """)
    results = cursor.fetchall()
    
    if not results:
        return "No active recommendations found."
    
    output = "💡 Security Recommendations:\n\n"
    for row in results:
        priority_emoji = {'P1': '🔴', 'P2': '🟡', 'P3': '🟢'}.get(row['priority'], '⚪')
        output += f"{priority_emoji} [{row['priority']}] {row['recommender']}\n"
        output += f"   {row['description'][:150]}...\n\n"
    
    return output

def _query_org_policies(cursor, params: Dict) -> str:
    """Query organization policies"""
    cursor.execute("""
        SELECT constraint_name, list_policy, boolean_policy, update_time
        FROM org_policies
        ORDER BY constraint_name
    """)
    results = cursor.fetchall()
    
    if not results:
        return "No organization policies found."
    
    output = f"📋 Organization Policies ({len(results)} configured):\n\n"
    for row in results:
        output += f"• {row['constraint_name']}\n"
        if row['boolean_policy']:
            output += f"  Boolean: {row['boolean_policy']}\n"
        if row['list_policy']:
            output += f"  List: {row['list_policy'][:100]}...\n"
        output += "\n"
    
    return output

def _query_service_usage(cursor, params: Dict) -> str:
    """Query service usage"""
    cursor.execute("""
        SELECT name, state
        FROM services
        WHERE state = 'ENABLED'
        ORDER BY name
    """)
    results = cursor.fetchall()
    
    if not results:
        return "No enabled services found."
    
    output = f"⚙️ Enabled Services ({len(results)} total):\n\n"
    
    # Group by service type
    security_services = []
    compute_services = []
    storage_services = []
    other_services = []
    
    for row in results:
        name = row['name']
        if 'security' in name or 'iam' in name or 'access' in name:
            security_services.append(name)
        elif 'compute' in name or 'container' in name or 'run' in name:
            compute_services.append(name)
        elif 'storage' in name or 'sql' in name or 'database' in name:
            storage_services.append(name)
        else:
            other_services.append(name)
    
    if security_services:
        output += f"🔒 Security Services ({len(security_services)}):\n"
        for svc in security_services[:5]:
            output += f"  • {svc}\n"
    
    if compute_services:
        output += f"\n💻 Compute Services ({len(compute_services)}):\n"
        for svc in compute_services[:5]:
            output += f"  • {svc}\n"
    
    if storage_services:
        output += f"\n💾 Storage Services ({len(storage_services)}):\n"
        for svc in storage_services[:5]:
            output += f"  • {svc}\n"
    
    return output

def _query_monitoring(cursor, params: Dict) -> str:
    """Query monitoring data"""
    cursor.execute("""
        SELECT name, description, enabled, notification_channels
        FROM alert_policies
        WHERE enabled = 1
    """)
    results = cursor.fetchall()
    
    output = "📊 Monitoring Configuration:\n\n"
    
    if results:
        output += f"Active Alert Policies: {len(results)}\n\n"
        for row in results[:5]:
            output += f"• {row['name']}\n"
            output += f"  {row['description'][:100] if row['description'] else 'No description'}\n"
            if row['notification_channels']:
                output += f"  Channels: {row['notification_channels']}\n"
            output += "\n"
    else:
        output += "⚠️ No active alert policies found.\n"
    
    return output

def _query_logs(cursor, params: Dict) -> str:
    """Query audit logs summary"""
    cursor.execute("""
        SELECT severity, resource_type, COUNT(*) as count
        FROM logs
        WHERE timestamp > datetime('now', '-7 days')
        GROUP BY severity, resource_type
        ORDER BY count DESC
        LIMIT 10
    """)
    results = cursor.fetchall()
    
    if not results:
        return "No recent audit logs found."
    
    output = "📝 Audit Logs (Last 7 Days):\n\n"
    for row in results:
        output += f"• {row['severity']} - {row['resource_type']}: {row['count']} events\n"
    
    return output

def _get_security_summary(cursor, params: Dict) -> str:
    """Get a comprehensive security summary highlighting the most critical issues"""
    output = "🚨 **SECURITY SUMMARY - MOST GLARING ISSUES** 🚨\n\n"
    
    critical_issues = []
    high_issues = []
    medium_issues = []
    
    # 1. Check for overly permissive firewall rules (CRITICAL)
    try:
        cursor.execute("""
            SELECT name, direction, source_ranges, allowed
            FROM firewall_rules
            WHERE source_ranges LIKE '%0.0.0.0/0%' 
            AND direction = 'INGRESS'
            AND disabled = 0
        """)
        open_rules = cursor.fetchall()
        if open_rules:
            for rule in open_rules:
                critical_issues.append({
                    'type': 'FIREWALL',
                    'title': f"🔥 Open Firewall Rule: {rule['name']}",
                    'detail': f"Allows traffic from 0.0.0.0/0 (entire internet) - {rule['allowed']}",
                    'remediation': "Restrict source_ranges to specific IPs or CIDR blocks"
                })
    except:
        pass
    
    # 2. Check for public storage buckets (CRITICAL)
    try:
        cursor.execute("""
            SELECT name, public_access, versioning_enabled, uniform_bucket_level_access
            FROM storage_buckets
            WHERE public_access != 'private' OR public_access IS NULL
        """)
        public_buckets = cursor.fetchall()
        if public_buckets:
            for bucket in public_buckets:
                severity = 'CRITICAL' if not bucket['versioning_enabled'] else 'HIGH'
                issue = {
                    'type': 'STORAGE',
                    'title': f"🪣 Public Storage Bucket: {bucket['name']}",
                    'detail': f"Bucket is publicly accessible. Versioning: {bucket['versioning_enabled']}, Uniform Access: {bucket['uniform_bucket_level_access']}",
                    'remediation': "Enable uniform bucket-level access, remove public access, enable versioning"
                }
                if severity == 'CRITICAL':
                    critical_issues.append(issue)
                else:
                    high_issues.append(issue)
    except:
        pass
    
    # 3. Check for critical security findings (CRITICAL/HIGH)
    try:
        cursor.execute("""
            SELECT category, severity, resource_name, description
            FROM security_findings
            WHERE severity IN ('CRITICAL', 'HIGH')
            ORDER BY CASE severity 
                     WHEN 'CRITICAL' THEN 1 
                     WHEN 'HIGH' THEN 2 
                     ELSE 3 END
            LIMIT 5
        """)
        findings = cursor.fetchall()
        for finding in findings:
            issue = {
                'type': 'FINDING',
                'title': f"🔍 {finding['severity']} Security Finding: {finding['category']}",
                'detail': f"Resource: {finding['resource_name']} - {finding['description'][:100]}...",
                'remediation': "Address immediately based on finding type"
            }
            if finding['severity'] == 'CRITICAL':
                critical_issues.append(issue)
            else:
                high_issues.append(issue)
    except:
        pass
    
    # 4. Check for service accounts with potential issues (HIGH)
    try:
        cursor.execute("""
            SELECT email, display_name
            FROM iam_accounts
            WHERE email LIKE '%iam.gserviceaccount.com%'
        """)
        service_accounts = cursor.fetchall()
        if len(service_accounts) > 5:
            high_issues.append({
                'type': 'IAM',
                'title': f"👤 Excessive Service Accounts: {len(service_accounts)} found",
                'detail': "Large number of service accounts increases attack surface",
                'remediation': "Audit and remove unused service accounts, use Workload Identity"
            })
    except:
        pass
    
    # 5. Check for missing monitoring (MEDIUM)
    try:
        cursor.execute("SELECT COUNT(*) as count FROM alert_policies WHERE enabled = 1")
        result = cursor.fetchone()
        if result['count'] == 0:
            medium_issues.append({
                'type': 'MONITORING',
                'title': "📊 No Active Alert Policies",
                'detail': "No monitoring alerts configured for security events",
                'remediation': "Set up alerts for failed logins, privilege escalation, unusual API usage"
            })
    except:
        pass
    
    # 6. Check for secrets without proper management (HIGH)
    try:
        cursor.execute("SELECT COUNT(*) as count FROM secrets")
        result = cursor.fetchone()
        if result['count'] > 0:
            cursor.execute("SELECT name FROM secrets")
            secrets = cursor.fetchall()
            for secret in secrets:
                high_issues.append({
                    'type': 'SECRETS',
                    'title': f"🔐 Secret Requires Review: {secret['name']}",
                    'detail': "Ensure proper access controls and rotation policies",
                    'remediation': "Enable automatic rotation, use IAM bindings for access control"
                })
    except:
        pass
    
    # Format output with priority
    if critical_issues:
        output += "🔴 **CRITICAL ISSUES (Fix Immediately)**\n"
        output += f"Found {len(critical_issues)} critical security issues requiring immediate attention:\n\n"
        for i, issue in enumerate(critical_issues, 1):
            output += f"{i}. {issue['title']}\n"
            output += f"   ➤ {issue['detail']}\n"
            output += f"   ✅ FIX: {issue['remediation']}\n\n"
    
    if high_issues:
        output += "🟡 **HIGH PRIORITY ISSUES (Fix Within 48 Hours)**\n"
        output += f"Found {len(high_issues)} high-priority issues:\n\n"
        for i, issue in enumerate(high_issues, 1):
            output += f"{i}. {issue['title']}\n"
            output += f"   ➤ {issue['detail']}\n"
            output += f"   ✅ FIX: {issue['remediation']}\n\n"
    
    if medium_issues:
        output += "🟢 **MEDIUM PRIORITY ISSUES (Schedule for This Week)**\n"
        output += f"Found {len(medium_issues)} medium-priority issues:\n\n"
        for i, issue in enumerate(medium_issues, 1):
            output += f"{i}. {issue['title']}\n"
            output += f"   ➤ {issue['detail']}\n"
            output += f"   ✅ FIX: {issue['remediation']}\n\n"
    
    # Summary statistics
    total_issues = len(critical_issues) + len(high_issues) + len(medium_issues)
    output += f"\n📈 **SUMMARY**\n"
    output += f"Total Security Issues: {total_issues}\n"
    output += f"• Critical: {len(critical_issues)}\n"
    output += f"• High: {len(high_issues)}\n"
    output += f"• Medium: {len(medium_issues)}\n\n"
    
    if total_issues == 0:
        output = "✅ **No Critical Security Issues Found!**\n\n"
        output += "Your GCP environment appears to be well-secured based on cached data.\n"
        output += "Continue with regular security audits and monitoring.\n"
    else:
        output += "⚡ **RECOMMENDED ACTION PLAN:**\n"
        output += "1. Address all CRITICAL issues immediately\n"
        output += "2. Schedule HIGH priority fixes within 48 hours\n"
        output += "3. Plan MEDIUM priority improvements for this week\n"
        output += "4. Enable continuous monitoring and alerting\n"
        output += "5. Implement regular security audits\n"
    
    return output

def _get_cache_status(cursor) -> str:
    """Get cache statistics"""
    tables = [
        'assets', 'security_findings', 'iam_policies', 'storage_buckets',
        'api_keys', 'recommendations', 'org_policies', 'services',
        'alert_policies', 'logs', 'firewall_rules', 'networks', 
        'compute_instances', 'databases', 'iam_accounts', 'secrets',
        'fetch_status', 'monitoring_metrics'
    ]
    
    output = "📊 Cache Status:\n\n"
    total_records = 0
    
    for table in tables:
        try:
            cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
            count = cursor.fetchone()['count']
            total_records += count
            if count > 0:
                output += f"• {table}: {count} records\n"
        except:
            pass
    
    output += f"\n📈 Total cached records: {total_records}\n"
    
    # Get last update time - check if cache_metadata exists first
    try:
        cursor.execute("""
            SELECT MAX(last_updated) as last_update
            FROM cache_metadata
        """)
        result = cursor.fetchone()
        if result and result['last_update']:
            output += f"🕐 Last refresh: {result['last_update']}\n"
    except:
        # Try fetch_status table as alternative
        try:
            cursor.execute("""
                SELECT MAX(fetched_at) as last_update
                FROM fetch_status
            """)
            result = cursor.fetchone()
            if result and result['last_update']:
                output += f"🕐 Last refresh: {result['last_update']}\n"
        except:
            pass  # No timestamp available
    
    return output

def _execute_custom_query(cursor, params: Dict) -> str:
    """Execute custom SQL query (with safety checks)"""
    sql = params.get('sql', '')
    
    if not sql:
        return "❌ No SQL query provided. Use {'sql': 'YOUR QUERY'}"
    
    # Basic safety checks
    dangerous_keywords = ['DROP', 'DELETE', 'UPDATE', 'INSERT', 'ALTER', 'CREATE']
    sql_upper = sql.upper()
    for keyword in dangerous_keywords:
        if keyword in sql_upper:
            return f"❌ Unsafe query detected. Only SELECT queries are allowed."
    
    try:
        cursor.execute(sql)
        results = cursor.fetchall()
        
        if not results:
            return "Query executed successfully but returned no results."
        
        # Format results as table
        columns = [description[0] for description in cursor.description]
        output = f"Query Results ({len(results)} rows):\n\n"
        
        # Show column headers
        output += " | ".join(columns) + "\n"
        output += "-" * (len(columns) * 15) + "\n"
        
        # Show first 10 rows
        for row in results[:10]:
            row_values = [str(row[col])[:30] for col in columns]
            output += " | ".join(row_values) + "\n"
        
        if len(results) > 10:
            output += f"\n... and {len(results) - 10} more rows"
        
        return output
        
    except Exception as e:
        return f"❌ Query error: {str(e)}"

def _query_firewall_rules(cursor, params: Dict) -> str:
    """Query firewall rules"""
    rule_name = params.get('rule_name', '')
    
    if rule_name:
        cursor.execute("""
            SELECT * FROM firewall_rules
            WHERE name = ?
        """, (rule_name,))
        result = cursor.fetchone()
        
        if not result:
            return f"Firewall rule {rule_name} not found."
        
        output = f"🔥 Firewall Rule: {rule_name}\n\n"
        output += f"Direction: {result['direction']}\n"
        output += f"Priority: {result['priority']}\n"
        output += f"Network: {result['network']}\n"
        output += f"Source Ranges: {result['source_ranges']}\n"
        if result['destination_ranges']:
            output += f"Destination Ranges: {result['destination_ranges']}\n"
        output += f"Allowed: {result['allowed']}\n"
        if result['denied']:
            output += f"Denied: {result['denied']}\n"
        output += f"Disabled: {'Yes' if result['disabled'] else 'No'}\n"
    else:
        cursor.execute("""
            SELECT name, direction, source_ranges, allowed, disabled
            FROM firewall_rules
            ORDER BY priority
        """)
        results = cursor.fetchall()
        
        if not results:
            return "No firewall rules found."
        
        output = f"🔥 Firewall Rules ({len(results)} total):\n\n"
        
        # Identify potential issues
        overly_permissive = []
        disabled_rules = []
        
        for rule in results:
            if '0.0.0.0/0' in str(rule['source_ranges']) and rule['direction'] == 'INGRESS':
                overly_permissive.append(rule)
            if rule['disabled']:
                disabled_rules.append(rule)
        
        if overly_permissive:
            output += f"⚠️ Overly Permissive Rules ({len(overly_permissive)}):\n"
            for rule in overly_permissive:
                output += f"  • {rule['name']}: Allows from 0.0.0.0/0 - {rule['allowed']}\n"
            output += "\n"
        
        if disabled_rules:
            output += f"🔒 Disabled Rules ({len(disabled_rules)}):\n"
            for rule in disabled_rules:
                output += f"  • {rule['name']}\n"
            output += "\n"
        
        output += "📋 All Rules:\n"
        for rule in results[:10]:  # Show first 10
            status = "🔓 OPEN" if '0.0.0.0/0' in str(rule['source_ranges']) else "✅"
            output += f"  {status} {rule['name']} ({rule['direction']}): {rule['source_ranges']}\n"
        
        if len(results) > 10:
            output += f"\n... and {len(results) - 10} more rules"
    
    return output

def _query_networks(cursor, params: Dict) -> str:
    """Query VPC networks"""
    cursor.execute("""
        SELECT * FROM networks
    """)
    results = cursor.fetchall()
    
    if not results:
        return "No VPC networks found."
    
    output = f"🌐 VPC Networks ({len(results)} total):\n\n"
    for row in results:
        output += f"• {row['name']}\n"
        if 'auto_create_subnetworks' in row:
            output += f"  Auto-create subnets: {row['auto_create_subnetworks']}\n"
        if 'description' in row and row['description']:
            output += f"  Description: {row['description']}\n"
        output += "\n"
    
    return output

def _query_compute_instances(cursor, params: Dict) -> str:
    """Query compute instances"""
    instance_name = params.get('instance_name', '')
    
    cursor.execute("""
        SELECT COUNT(*) as count FROM compute_instances
    """)
    result = cursor.fetchone()
    
    if result['count'] == 0:
        # No compute instances in this table, check assets table instead
        cursor.execute("""
            SELECT name, location, labels, create_time
            FROM assets
            WHERE asset_type = 'compute.googleapis.com/Instance'
            ORDER BY create_time DESC
        """)
        results = cursor.fetchall()
        
        if not results:
            return "No compute instances found."
        
        output = f"💻 Compute Instances ({len(results)} total):\n\n"
        for instance in results[:10]:
            output += f"• {instance['name']}\n"
            output += f"  Location: {instance['location']}\n"
            if instance['labels']:
                output += f"  Labels: {instance['labels']}\n"
            output += "\n"
        
        if len(results) > 10:
            output += f"... and {len(results) - 10} more instances"
    else:
        output = "Compute instances data available but needs schema check."
    
    return output

def _query_databases(cursor, params: Dict) -> str:
    """Query database instances"""
    cursor.execute("""
        SELECT COUNT(*) as count FROM databases
    """)
    result = cursor.fetchone()
    
    if result['count'] == 0:
        # Check assets table for database instances
        cursor.execute("""
            SELECT name, asset_type, location
            FROM assets
            WHERE asset_type LIKE '%sql%' OR asset_type LIKE '%database%' OR asset_type LIKE '%spanner%'
            ORDER BY asset_type
        """)
        results = cursor.fetchall()
        
        if not results:
            return "No database instances found."
        
        output = f"🗄️ Database Instances ({len(results)} total):\n\n"
        
        # Group by type
        by_type = {}
        for db in results:
            db_type = db['asset_type'].split('/')[-1]
            if db_type not in by_type:
                by_type[db_type] = []
            by_type[db_type].append(db)
        
        for db_type, instances in by_type.items():
            output += f"{db_type} ({len(instances)}):\n"
            for instance in instances[:3]:
                output += f"  • {instance['name']} ({instance['location']})\n"
            if len(instances) > 3:
                output += f"  ... and {len(instances) - 3} more\n"
            output += "\n"
    else:
        output = "Database instances data available but needs schema check."
    
    return output

def _query_iam_accounts(cursor, params: Dict) -> str:
    """Query IAM service accounts and users"""
    account_email = params.get('email', '')
    
    if account_email:
        cursor.execute("""
            SELECT * FROM iam_accounts
            WHERE email = ?
        """, (account_email,))
        result = cursor.fetchone()
        
        if not result:
            return f"IAM account {account_email} not found."
        
        output = f"👤 IAM Account: {account_email}\n\n"
        for key in result.keys():
            if result[key]:
                output += f"{key}: {result[key]}\n"
    else:
        cursor.execute("""
            SELECT * FROM iam_accounts
        """)
        results = cursor.fetchall()
        
        if not results:
            return "No IAM accounts found in cache."
        
        output = f"👥 IAM Accounts ({len(results)} total):\n\n"
        
        # Separate service accounts and users
        service_accounts = []
        user_accounts = []
        
        for account in results:
            if 'email' in account:
                email = account['email']
                if 'iam.gserviceaccount.com' in email:
                    service_accounts.append(account)
                else:
                    user_accounts.append(account)
        
        if service_accounts:
            output += f"🤖 Service Accounts ({len(service_accounts)}):\n"
            for sa in service_accounts[:5]:
                output += f"  • {sa['email']}\n"
                if 'display_name' in sa and sa['display_name']:
                    output += f"    Name: {sa['display_name']}\n"
            if len(service_accounts) > 5:
                output += f"  ... and {len(service_accounts) - 5} more\n"
            output += "\n"
        
        if user_accounts:
            output += f"👤 User Accounts ({len(user_accounts)}):\n"
            for user in user_accounts[:5]:
                output += f"  • {user['email']}\n"
            if len(user_accounts) > 5:
                output += f"  ... and {len(user_accounts) - 5} more\n"
    
    return output

def _query_secrets(cursor, params: Dict) -> str:
    """Query secrets from Secret Manager"""
    secret_name = params.get('secret_name', '')
    
    if secret_name:
        cursor.execute("""
            SELECT * FROM secrets
            WHERE name LIKE ?
        """, (f'%{secret_name}%',))
        results = cursor.fetchall()
        
        if not results:
            return f"No secrets matching '{secret_name}' found."
        
        output = f"🔐 Secrets matching '{secret_name}':\n\n"
        for secret in results:
            output += f"• {secret['name']}\n"
            for key in secret.keys():
                if key != 'name' and secret[key]:
                    output += f"  {key}: {secret[key]}\n"
            output += "\n"
    else:
        cursor.execute("""
            SELECT * FROM secrets
        """)
        results = cursor.fetchall()
        
        if not results:
            return "No secrets found in Secret Manager."
        
        output = f"🔐 Secrets ({len(results)} total):\n\n"
        
        for secret in results:
            output += f"• {secret['name']}\n"
            if 'create_time' in secret:
                output += f"  Created: {secret['create_time']}\n"
            if 'state' in secret:
                output += f"  State: {secret['state']}\n"
            output += "\n"
    
    return output

def _query_msa_analysis(cursor, params: Dict) -> str:
    """Query MSA (Monthly Service Announcement) analysis results"""
    
    # Check if we have any MSA data
    cursor.execute("SELECT COUNT(*) as count FROM msa_emails")
    result = cursor.fetchone()
    total_msas = result['count'] if result else 0
    
    if total_msas == 0:
        return "No MSA analyses found. Upload an MSA email through the MSA Analyzer to see results."
    
    output = f"📧 MSA Analysis History ({total_msas} emails analyzed):\n\n"
    
    # Get recent MSA analyses
    cursor.execute("""
        SELECT 
            id,
            project_id,
            analyzed_date,
            (SELECT COUNT(*) FROM msa_changes WHERE msa_email_id = msa_emails.id) as change_count
        FROM msa_emails
        ORDER BY analyzed_date DESC
        LIMIT 5
    """)
    
    recent_msas = cursor.fetchall()
    
    for msa in recent_msas:
        output += f"MSA #{msa['id']} - Analyzed: {msa['analyzed_date']}\n"
        output += f"  Project: {msa['project_id'] or 'All projects'}\n"
        output += f"  Changes detected: {msa['change_count']}\n\n"
    
    # Get summary statistics
    cursor.execute("""
        SELECT 
            impact_level,
            COUNT(*) as count
        FROM msa_changes
        GROUP BY impact_level
    """)
    
    impact_stats = cursor.fetchall()
    if impact_stats:
        output += "📊 Overall Impact Distribution:\n"
        for stat in impact_stats:
            emoji = "🔴" if stat['impact_level'] == 'critical' else "🟠" if stat['impact_level'] == 'high' else "🟡" if stat['impact_level'] == 'medium' else "🟢"
            output += f"  {emoji} {stat['impact_level'].upper()}: {stat['count']} changes\n"
    
    return output

def _query_msa_changes(cursor, params: Dict) -> str:
    """Query specific MSA changes with structured data"""
    
    # Build query based on parameters
    where_clauses = []
    query_params = []
    
    if params.get('service'):
        where_clauses.append("service LIKE ?")
        query_params.append(f"%{params['service']}%")
    
    if params.get('impact_level'):
        where_clauses.append("impact_level = ?")
        query_params.append(params['impact_level'])
    
    if params.get('msa_id'):
        where_clauses.append("msa_email_id = ?")
        query_params.append(params['msa_id'])
    
    if params.get('permission'):
        # Search for specific permission in old_permission or new_permissions
        where_clauses.append("(old_permission LIKE ? OR new_permissions LIKE ?)")
        query_params.extend([f"%{params['permission']}%", f"%{params['permission']}%"])
    
    where_clause = " WHERE " + " AND ".join(where_clauses) if where_clauses else ""
    
    query = f"""
        SELECT * FROM msa_changes
        {where_clause}
        ORDER BY 
            CASE impact_level
                WHEN 'critical' THEN 1
                WHEN 'high' THEN 2
                WHEN 'medium' THEN 3
                WHEN 'low' THEN 4
                ELSE 5
            END,
            effective_date ASC
        LIMIT 20
    """
    
    cursor.execute(query, query_params)
    changes = cursor.fetchall()
    
    if not changes:
        return "No MSA changes found matching your criteria."
    
    output = f"🔄 MSA Changes ({len(changes)} found):\n\n"
    
    for change in changes:
        # Determine emoji based on impact level
        if change['impact_level'] == 'critical':
            emoji = "🔴"
        elif change['impact_level'] == 'high':
            emoji = "🟠"
        elif change['impact_level'] == 'medium':
            emoji = "🟡"
        else:
            emoji = "🟢"
        
        output += f"{emoji} {change['service']} - {change['change_type']}\n"
        output += f"   {change['description'][:100]}...\n" if len(change['description']) > 100 else f"   {change['description']}\n"
        
        # Show structured permission data if available
        if change['old_permission']:
            output += f"   🔐 Old Permission: `{change['old_permission']}`\n"
        
        if change['new_permissions']:
            try:
                import json
                new_perms = json.loads(change['new_permissions']) if isinstance(change['new_permissions'], str) else change['new_permissions']
                if new_perms:
                    output += f"   🆕 New Permissions: {', '.join([f'`{p}`' for p in new_perms])}\n"
            except:
                pass
        
        # Show API parameters if available
        if change['api_parameters']:
            try:
                import json
                api_params = json.loads(change['api_parameters']) if isinstance(change['api_parameters'], str) else change['api_parameters']
                if api_params:
                    output += f"   🔧 API Parameters:\n"
                    for param_name, param_info in api_params.items():
                        if isinstance(param_info, dict) and 'values' in param_info:
                            output += f"      • {param_name}: {', '.join(param_info['values'])}\n"
            except:
                pass
        
        if change['effective_date']:
            output += f"   📅 Effective: {change['effective_date']}\n"
        
        if change['required_action']:
            output += f"   ⚡ Action: {change['required_action'][:100]}...\n" if len(change['required_action']) > 100 else f"   ⚡ Action: {change['required_action']}\n"
        
        if change['affects_predefined_roles'] is False:
            output += f"   ✅ Predefined roles are NOT affected\n"
        
        if change['testing_available']:
            output += f"   🧪 Testing available for early validation\n"
        
        output += "\n"
    
    return output

def _query_msa_impact(cursor, params: Dict) -> str:
    """Query MSA impact assessments for specific projects"""
    
    project_id = params.get('project_id')
    
    if not project_id:
        # Get overall impact summary
        cursor.execute("""
            SELECT 
                project_id,
                COUNT(*) as assessment_count,
                SUM(resource_count) as total_resources
            FROM msa_impact_assessments
            GROUP BY project_id
        """)
        
        results = cursor.fetchall()
        
        if not results:
            return "No MSA impact assessments found. Analyze an MSA with a specific project ID to see impact."
        
        output = "🎯 MSA Impact Summary by Project:\n\n"
        
        for row in results:
            output += f"Project: {row['project_id']}\n"
            output += f"  Assessments: {row['assessment_count']}\n"
            output += f"  Total resources affected: {row['total_resources']}\n\n"
        
        return output
    
    # Get impact for specific project
    cursor.execute("""
        SELECT 
            ia.*,
            c.service,
            c.change_type,
            c.description
        FROM msa_impact_assessments ia
        JOIN msa_changes c ON ia.msa_change_id = c.id
        WHERE ia.project_id = ?
        ORDER BY ia.resource_count DESC
    """, (project_id,))
    
    assessments = cursor.fetchall()
    
    if not assessments:
        return f"No MSA impact assessments found for project: {project_id}"
    
    output = f"🎯 MSA Impact Assessment for {project_id}:\n\n"
    
    total_resources = sum(a['resource_count'] for a in assessments)
    output += f"📊 Total resources affected: {total_resources}\n\n"
    
    for assessment in assessments[:10]:  # Show top 10
        emoji = "🔴" if assessment['impact_level'] == 'critical' else "🟠" if assessment['impact_level'] == 'high' else "🟡"
        
        output += f"{emoji} {assessment['service']} - {assessment['change_type']}\n"
        output += f"   Resource type: {assessment['resource_type']}\n"
        output += f"   Resources affected: {assessment['resource_count']}\n"
        
        # Parse and show recommendations if available
        if assessment['recommended_actions']:
            try:
                actions = eval(assessment['recommended_actions']) if isinstance(assessment['recommended_actions'], str) else assessment['recommended_actions']
                if actions and isinstance(actions, list):
                    output += "   Recommended actions:\n"
                    for action in actions[:3]:  # Show first 3
                        output += f"     • {action}\n"
            except:
                pass
        
        output += "\n"
    
    if len(assessments) > 10:
        output += f"... and {len(assessments) - 10} more assessments"
    
    return output

def _query_msa_permissions(cursor, params: Dict) -> str:
    """Query MSA permission changes with detailed mapping"""
    
    # Check for specific permission query
    permission_name = params.get('permission', '')
    
    if permission_name:
        # Search for a specific permission
        cursor.execute("""
            SELECT * FROM msa_changes
            WHERE (old_permission LIKE ? OR new_permissions LIKE ?)
            AND (old_permission IS NOT NULL OR new_permissions IS NOT NULL)
            ORDER BY effective_date ASC
        """, (f"%{permission_name}%", f"%{permission_name}%"))
    else:
        # Get all permission changes
        cursor.execute("""
            SELECT * FROM msa_changes
            WHERE old_permission IS NOT NULL OR new_permissions IS NOT NULL
            ORDER BY service, effective_date ASC
        """)
    
    changes = cursor.fetchall()
    
    if not changes:
        if permission_name:
            return f"No MSA changes found for permission: {permission_name}"
        else:
            return "No permission-related MSA changes found. Upload an MSA with permission changes to see results."
    
    if permission_name:
        output = f"🔐 MSA Permission Changes for '{permission_name}':\n\n"
    else:
        output = f"🔐 All MSA Permission Changes ({len(changes)} found):\n\n"
    
    # Group by service for better organization
    by_service = {}
    for change in changes:
        service = change['service']
        if service not in by_service:
            by_service[service] = []
        by_service[service].append(change)
    
    for service, service_changes in by_service.items():
        output += f"📦 {service}:\n"
        
        for change in service_changes:
            output += f"\n  📝 {change['change_type'].replace('_', ' ').title()}\n"
            
            # Show the permission mapping
            if change['old_permission']:
                output += f"    FROM: `{change['old_permission']}`\n"
                
                if change['new_permissions']:
                    try:
                        import json
                        new_perms = json.loads(change['new_permissions']) if isinstance(change['new_permissions'], str) else change['new_permissions']
                        if new_perms:
                            output += f"    TO:   "
                            for i, perm in enumerate(new_perms):
                                if i == 0:
                                    output += f"`{perm}`\n"
                                else:
                                    output += f"          `{perm}`\n"
                    except:
                        pass
            
            # Show what the permission change means
            if change['description']:
                # Extract the key information
                desc_lines = change['description'].split('. ')
                for line in desc_lines[:2]:  # Show first 2 sentences
                    if 'currently' in line.lower() or 'after' in line.lower() or 'will' in line.lower():
                        output += f"    ℹ️  {line}.\n"
            
            # Show effective date
            if change['effective_date']:
                output += f"    📅 Effective: {change['effective_date']}\n"
            
            # Show required action
            if change['required_action']:
                output += f"    ⚡ Action: {change['required_action'][:80]}...\n" if len(change['required_action']) > 80 else f"    ⚡ Action: {change['required_action']}\n"
            
            # Show if testing is available
            if change['testing_available']:
                output += f"    🧪 Early testing available\n"
            
            # Show API parameters if relevant
            if change['api_parameters']:
                try:
                    import json
                    api_params = json.loads(change['api_parameters']) if isinstance(change['api_parameters'], str) else change['api_parameters']
                    if api_params:
                        output += f"    🔧 Related API parameters:\n"
                        for param_name, param_info in api_params.items():
                            if isinstance(param_info, dict) and 'permissions_required' in param_info:
                                output += f"       • {param_name}:\n"
                                for value, perms in param_info['permissions_required'].items():
                                    output += f"         - {value}: requires {', '.join(perms)}\n"
                except:
                    pass
        
        output += "\n"
    
    # Add summary
    output += "📊 Summary:\n"
    output += f"  • Total permission changes: {len(changes)}\n"
    output += f"  • Services affected: {', '.join(by_service.keys())}\n"
    
    # Find earliest effective date
    effective_dates = [c['effective_date'] for c in changes if c['effective_date']]
    if effective_dates:
        earliest = min(effective_dates)
        output += f"  • Earliest change date: {earliest}\n"
    
    output += "\n💡 Tip: Query specific permissions like 'bigquery.datasets.get' to see detailed mapping."
    
    return output


def _query_knowledge_base(cursor, params: Dict) -> str:
    """Query all knowledge base information"""
    output = "📚 Knowledge Base Overview\n" + "=" * 40 + "\n\n"
    
    # Get counts from each table
    tables = [
        ('enterprise_policies', 'Enterprise Policies'),
        ('coding_standards', 'Coding Standards'),
        ('compliance_frameworks', 'Compliance Requirements'),
        ('best_practices', 'Best Practices')
    ]
    
    for table, name in tables:
        cursor.execute(f"SELECT COUNT(*) as count FROM {table}")
        count = cursor.fetchone()['count']
        output += f"• {name}: {count} entries\n"
    
    output += "\n📋 Quick Access:\n"
    output += "• Query 'coding_standards' for development guidelines\n"
    output += "• Query 'enterprise_policies' for security policies\n"
    output += "• Query 'best_practices' for GCP recommendations\n"
    output += "• Query 'compliance' for regulatory requirements\n"
    
    # Show sample entries
    output += "\n🔍 Recent Additions:\n"
    cursor.execute("""
        SELECT 'Policy' as type, policy_name as name, severity 
        FROM enterprise_policies 
        ORDER BY created_at DESC LIMIT 2
    """)
    for row in cursor.fetchall():
        output += f"  [{row['severity']}] {row['name']}\n"
    
    cursor.execute("""
        SELECT 'Standard' as type, standard_name as name, severity
        FROM coding_standards
        ORDER BY created_at DESC LIMIT 2
    """)
    for row in cursor.fetchall():
        output += f"  [{row['severity']}] {row['name']}\n"
    
    return output


def _query_coding_standards(cursor, params: Dict) -> str:
    """Query coding standards and test requirements"""
    output = "📝 Coding Standards & Test Requirements\n" + "=" * 40 + "\n\n"
    
    language = params.get('language', 'Python')
    severity = params.get('severity')
    search = params.get('search')
    
    # Build query
    query = "SELECT * FROM coding_standards WHERE is_active = 1"
    query_params = []
    
    if language:
        query += " AND language = ?"
        query_params.append(language)
    
    if severity:
        query += " AND severity = ?"
        query_params.append(severity)
    
    if search:
        query += " AND (standard_name LIKE ? OR rule_description LIKE ? OR tags LIKE ?)"
        query_params.extend([f"%{search}%", f"%{search}%", f"%{search}%"])
    
    cursor.execute(query, query_params)
    standards = cursor.fetchall()
    
    if not standards:
        return f"No coding standards found for language='{language}'"
    
    # Group by severity
    by_severity = {'ERROR': [], 'WARNING': [], 'INFO': []}
    for standard in standards:
        by_severity[standard['severity']].append(standard)
    
    # Show ERROR level first
    if by_severity['ERROR']:
        output += "🔴 ERROR Standards (Must Fix):\n"
        for std in by_severity['ERROR']:
            output += f"\n  📌 {std['standard_name']}\n"
            output += f"     Rule: {std['rule_description']}\n"
            if std['example_good']:
                output += f"     ✅ Good: {std['example_good'][:100]}\n"
            if std['example_bad']:
                output += f"     ❌ Bad: {std['example_bad'][:100]}\n"
            if std['linter_rule']:
                output += f"     🔧 Linter: {std['linter_rule']}\n"
    
    # Show WARNING level
    if by_severity['WARNING']:
        output += "\n⚠️ WARNING Standards (Should Fix):\n"
        for std in by_severity['WARNING']:
            output += f"\n  📌 {std['standard_name']}\n"
            output += f"     Rule: {std['rule_description']}\n"
            if std['example_good']:
                output += f"     ✅ Good: {std['example_good'][:100]}\n"
    
    # Show INFO level
    if by_severity['INFO']:
        output += "\n💡 INFO Standards (Best Practices):\n"
        for std in by_severity['INFO']:
            output += f"\n  📌 {std['standard_name']}\n"
            output += f"     Rule: {std['rule_description']}\n"
    
    # Summary
    output += f"\n📊 Total: {len(standards)} standards for {language}\n"
    output += f"   • Errors: {len(by_severity['ERROR'])}\n"
    output += f"   • Warnings: {len(by_severity['WARNING'])}\n"
    output += f"   • Info: {len(by_severity['INFO'])}\n"
    
    # Test-specific standards
    test_standards = [s for s in standards if 'test' in s['standard_name'].lower()]
    if test_standards:
        output += f"\n🧪 Test Standards: {len(test_standards)} found\n"
        for std in test_standards:
            output += f"   • {std['standard_name']}\n"
    
    return output


def _query_enterprise_policies(cursor, params: Dict) -> str:
    """Query enterprise security and governance policies"""
    output = "🛡️ Enterprise Security Policies\n" + "=" * 40 + "\n\n"
    
    category = params.get('category')
    severity = params.get('severity')
    
    # Build query
    query = "SELECT * FROM enterprise_policies WHERE is_active = 1"
    query_params = []
    
    if category:
        query += " AND category = ?"
        query_params.append(category)
    
    if severity:
        query += " AND severity = ?"
        query_params.append(severity)
    
    query += " ORDER BY severity DESC, policy_name"
    
    cursor.execute(query, query_params)
    policies = cursor.fetchall()
    
    if not policies:
        return "No active enterprise policies found"
    
    # Group by category
    by_category = {}
    for policy in policies:
        cat = policy['category']
        if cat not in by_category:
            by_category[cat] = []
        by_category[cat].append(policy)
    
    # Show policies by category
    for category, cat_policies in by_category.items():
        output += f"📁 {category}:\n"
        for policy in cat_policies:
            severity_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(policy['severity'], '⚪')
            output += f"\n  {severity_icon} {policy['policy_name']}\n"
            output += f"     {policy['description']}\n"
            if policy['implementation_guide']:
                guide_lines = policy['implementation_guide'].split('\\n')[:2]
                for line in guide_lines:
                    if line.strip():
                        output += f"     → {line.strip()}\n"
            if policy['exceptions']:
                output += f"     ⚠️ Exceptions: {policy['exceptions'][:100]}\n"
    
    # Summary
    output += f"\n📊 Total: {len(policies)} active policies\n"
    severity_counts = {}
    for policy in policies:
        severity_counts[policy['severity']] = severity_counts.get(policy['severity'], 0) + 1
    for sev in ['CRITICAL', 'HIGH', 'MEDIUM', 'LOW']:
        if sev in severity_counts:
            output += f"   • {sev}: {severity_counts[sev]}\n"
    
    return output


def _query_best_practices(cursor, params: Dict) -> str:
    """Query GCP best practices and recommendations"""
    output = "✨ GCP Best Practices\n" + "=" * 40 + "\n\n"
    
    service = params.get('service')
    category = params.get('category')
    
    # Build query
    query = "SELECT * FROM best_practices WHERE is_active = 1"
    query_params = []
    
    if service:
        query += " AND service = ?"
        query_params.append(service)
    
    if category:
        query += " AND category = ?"
        query_params.append(category)
    
    cursor.execute(query, query_params)
    practices = cursor.fetchall()
    
    if not practices:
        return "No best practices found for the specified criteria"
    
    # Group by service
    by_service = {}
    for practice in practices:
        svc = practice['service']
        if svc not in by_service:
            by_service[svc] = []
        by_service[svc].append(practice)
    
    # Show practices by service
    for service, svc_practices in by_service.items():
        output += f"☁️ {service}:\n"
        for practice in svc_practices:
            output += f"\n  📌 {practice['practice_name']}\n"
            output += f"     Category: {practice['category']}\n"
            output += f"     Rationale: {practice['rationale']}\n"
            
            # Show implementation
            if practice['gcloud_command']:
                output += f"     🔧 Command: {practice['gcloud_command']}\n"
            elif practice['terraform_snippet']:
                snippet = practice['terraform_snippet'][:100]
                output += f"     📄 Terraform: {snippet}...\n"
            else:
                guide = practice['implementation_guide'][:100]
                output += f"     📝 How: {guide}...\n"
            
            # Show risk
            if practice['risk_if_not_followed']:
                output += f"     ⚠️ Risk: {practice['risk_if_not_followed']}\n"
    
    # Summary
    output += f"\n📊 Total: {len(practices)} best practices\n"
    output += f"   • Services: {', '.join(by_service.keys())}\n"
    
    # Categories
    categories = set(p['category'] for p in practices)
    output += f"   • Categories: {', '.join(categories)}\n"
    
    return output


def _query_compliance(cursor, params: Dict) -> str:
    """Query compliance framework requirements"""
    output = "📋 Compliance Framework Requirements\n" + "=" * 40 + "\n\n"
    
    framework = params.get('framework')
    status = params.get('status')
    
    # Build query
    query = "SELECT * FROM compliance_frameworks WHERE 1=1"
    query_params = []
    
    if framework:
        query += " AND framework_name = ?"
        query_params.append(framework)
    
    if status:
        query += " AND compliance_status = ?"
        query_params.append(status)
    
    cursor.execute(query, query_params)
    requirements = cursor.fetchall()
    
    if not requirements:
        return "No compliance requirements found"
    
    # Group by framework
    by_framework = {}
    for req in requirements:
        fw = req['framework_name']
        if fw not in by_framework:
            by_framework[fw] = []
        by_framework[fw].append(req)
    
    # Show requirements by framework
    for framework, fw_reqs in by_framework.items():
        output += f"📜 {framework}:\n"
        for req in fw_reqs:
            status_icon = {
                'COMPLIANT': '✅',
                'NON_COMPLIANT': '❌',
                'PARTIAL': '⚠️',
                'NOT_ASSESSED': '❓'
            }.get(req['compliance_status'], '⚪')
            
            output += f"\n  {status_icon} {req['requirement_id']}: {req['requirement_text']}\n"
            if req['description']:
                output += f"     {req['description'][:100]}\n"
            
            # Show GCP mapping
            if req['gcp_mapping']:
                try:
                    import json
                    services = json.loads(req['gcp_mapping'])
                    output += f"     🔗 GCP Services: {', '.join(services)}\n"
                except:
                    pass
            
            # Show remediation if non-compliant
            if req['compliance_status'] == 'NON_COMPLIANT' and req['remediation_steps']:
                output += f"     🔧 Fix: {req['remediation_steps'][:100]}\n"
    
    # Summary
    output += f"\n📊 Compliance Summary:\n"
    status_counts = {}
    for req in requirements:
        status_counts[req['compliance_status']] = status_counts.get(req['compliance_status'], 0) + 1
    
    for status, count in status_counts.items():
        output += f"   • {status}: {count}\n"
    
    # Frameworks
    frameworks = set(r['framework_name'] for r in requirements)
    output += f"   • Frameworks: {', '.join(frameworks)}\n"
    
    return output


def _query_context_aware_analysis(cursor, params: Dict) -> str:
    """
    Full feedback loop analysis connecting MSA changes with security findings, 
    assets, IAM policies, and remediation effectiveness.
    
    This creates a comprehensive context-aware view showing how changes in one
    area (like MSA announcements) ripple through the entire security posture.
    """
    
    # Parse parameters
    focus_area = params.get('focus', 'all') if params else 'all'
    timeframe = params.get('timeframe', '30_days') if params else '30_days'
    
    output = f"🔄 **Context-Aware Security Analysis** - Full Feedback Loop\n"
    output += f"{'=' * 60}\n\n"
    
    # 1. MSA Change Impact Analysis
    output += "## 🧠 MSA Impact Propagation\n\n"
    
    cursor.execute("""
        SELECT 
            mc.service,
            mc.change_type,
            mc.description,
            mc.effective_date,
            COUNT(DISTINCT ip.role) as affected_roles,
            COUNT(DISTINCT a.resource_name) as affected_assets
        FROM msa_changes mc
        LEFT JOIN iam_policies ip ON (
            mc.old_permission IS NOT NULL AND 
            ip.role LIKE '%' || LOWER(mc.service) || '%'
        )
        LEFT JOIN assets a ON a.asset_type LIKE '%' || LOWER(mc.service) || '%'
        GROUP BY mc.id, mc.service, mc.change_type
        ORDER BY affected_roles DESC, affected_assets DESC
        LIMIT 5
    """)
    
    msa_impacts = cursor.fetchall()
    
    for impact in msa_impacts:
        output += f"### 📧 {impact['service']} - {impact['change_type']}\n"
        output += f"**Impact Radius:**\n"
        output += f"- 🔐 IAM Roles Affected: {impact['affected_roles']}\n"
        output += f"- 🏗️ Assets Potentially Impacted: {impact['affected_assets']}\n"
        output += f"- 📅 Effective Date: {impact['effective_date']}\n"
        output += f"- 📝 Description: {impact['description'][:200]}...\n\n"
    
    # 2. Security Finding Correlation
    output += "## 🛡️ Security Findings Correlation\n\n"
    
    cursor.execute("""
        SELECT 
            sf.finding_class,
            sf.severity,
            COUNT(*) as finding_count,
            GROUP_CONCAT(DISTINCT sf.resource_name, ', ') as sample_resources
        FROM security_findings sf
        WHERE sf.state = 'ACTIVE'
        GROUP BY sf.finding_class, sf.severity
        ORDER BY 
            CASE sf.severity 
                WHEN 'CRITICAL' THEN 1 
                WHEN 'HIGH' THEN 2 
                WHEN 'MEDIUM' THEN 3 
                ELSE 4 
            END,
            finding_count DESC
        LIMIT 8
    """)
    
    findings = cursor.fetchall()
    
    for finding in findings:
        severity_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(finding['severity'], '⚪')
        output += f"{severity_icon} **{finding['finding_class']}** ({finding['severity']})\n"
        output += f"   • Count: {finding['finding_count']} active findings\n"
        
        # Show sample resources (truncated)
        if finding['sample_resources']:
            resources = finding['sample_resources'].split(', ')[:3]
            output += f"   • Sample Resources: {', '.join(resources)}\n"
            if len(resources) == 3:
                output += f"   • ... and more\n"
        output += "\n"
    
    # 3. Asset Vulnerability Cross-Analysis
    output += "## 🏗️ Asset-Security Cross-Analysis\n\n"
    
    cursor.execute("""
        SELECT 
            a.asset_type,
            COUNT(DISTINCT a.resource_name) as total_assets,
            COUNT(DISTINCT sf.resource_name) as assets_with_findings,
            ROUND(
                (CAST(COUNT(DISTINCT sf.resource_name) AS FLOAT) / 
                 CAST(COUNT(DISTINCT a.resource_name) AS FLOAT)) * 100, 1
            ) as vulnerability_rate
        FROM assets a
        LEFT JOIN security_findings sf ON a.resource_name = sf.resource_name
        GROUP BY a.asset_type
        HAVING total_assets > 0
        ORDER BY vulnerability_rate DESC, total_assets DESC
        LIMIT 10
    """)
    
    asset_analysis = cursor.fetchall()
    
    for asset in asset_analysis:
        vuln_rate = asset['vulnerability_rate'] or 0
        risk_icon = "🔴" if vuln_rate > 50 else "🟠" if vuln_rate > 20 else "🟡" if vuln_rate > 5 else "🟢"
        
        output += f"{risk_icon} **{asset['asset_type']}**\n"
        output += f"   • Total Assets: {asset['total_assets']}\n"
        output += f"   • With Security Findings: {asset['assets_with_findings']}\n"
        output += f"   • Vulnerability Rate: {vuln_rate}%\n\n"
    
    # 4. IAM Permission Risk Analysis
    output += "## 🔐 IAM Permission Risk Pattern\n\n"
    
    cursor.execute("""
        SELECT 
            ip.role,
            COUNT(DISTINCT ip.member) as member_count,
            COUNT(DISTINCT ip.resource_name) as resource_count,
            CASE 
                WHEN ip.role LIKE '%admin%' OR ip.role LIKE '%owner%' THEN 'HIGH_PRIVILEGE'
                WHEN ip.role LIKE '%viewer%' OR ip.role LIKE '%browser%' THEN 'READ_ONLY'
                WHEN ip.role LIKE 'projects/%/roles/%' THEN 'CUSTOM_ROLE'
                ELSE 'STANDARD_ROLE'
            END as risk_category
        FROM iam_policies ip
        GROUP BY ip.role
        ORDER BY member_count DESC, resource_count DESC
        LIMIT 12
    """)
    
    iam_risks = cursor.fetchall()
    
    # Group by risk category
    by_risk = {}
    for iam in iam_risks:
        category = iam['risk_category']
        if category not in by_risk:
            by_risk[category] = []
        by_risk[category].append(iam)
    
    for category, roles in by_risk.items():
        category_icon = {
            'HIGH_PRIVILEGE': '🔴',
            'CUSTOM_ROLE': '🟠', 
            'STANDARD_ROLE': '🟡',
            'READ_ONLY': '🟢'
        }.get(category, '⚪')
        
        output += f"{category_icon} **{category.replace('_', ' ').title()}**\n"
        for role in roles[:3]:  # Show top 3 in each category
            output += f"   • {role['role']}: {role['member_count']} members, {role['resource_count']} resources\n"
        output += "\n"
    
    # 5. Feedback Loop Recommendations
    output += "## 🎯 Context-Aware Recommendations\n\n"
    
    output += "**Immediate Actions:**\n"
    output += "1. 🔄 Review MSA changes affecting high-privilege roles\n"
    output += "2. 🛡️ Prioritize critical/high security findings on high-asset-count types\n"
    output += "3. 🔐 Audit custom roles with broad resource access\n"
    output += "4. 📊 Implement monitoring for vulnerability rate trends\n\n"
    
    output += "**Feedback Loop Optimization:**\n"
    output += "1. 📈 Establish baseline metrics for each asset type\n"
    output += "2. 🔔 Set up alerts when vulnerability rates exceed thresholds\n"
    output += "3. 🤖 Automate remediation for low-risk, high-volume findings\n"
    output += "4. 📋 Create periodic reviews linking MSA changes to security posture changes\n\n"
    
    return output


def _query_cross_impact_analysis(cursor, params: Dict) -> str:
    """
    Analyze how changes in one security domain affect other domains.
    This implements the full feedback loop showing cascading impacts.
    """
    
    # Parse parameters
    domain = params.get('domain', 'all') if params else 'all'
    depth = params.get('depth', 'medium') if params else 'medium'
    
    output = f"🌐 **Cross-Impact Analysis** - Security Domain Interactions\n"
    output += f"{'=' * 65}\n\n"
    
    # 1. MSA → IAM → Asset Impact Chain
    output += "## 🔗 Change Propagation Chain Analysis\n\n"
    
    cursor.execute("""
        WITH msa_service_impact AS (
            SELECT 
                mc.service,
                mc.change_type,
                COUNT(DISTINCT ip.role) as iam_roles_affected,
                COUNT(DISTINCT a.resource_name) as assets_potentially_affected
            FROM msa_changes mc
            LEFT JOIN iam_policies ip ON (
                LOWER(ip.role) LIKE '%' || LOWER(mc.service) || '%' OR
                LOWER(ip.resource_name) LIKE '%' || LOWER(mc.service) || '%'
            )
            LEFT JOIN assets a ON LOWER(a.asset_type) LIKE '%' || LOWER(mc.service) || '%'
            GROUP BY mc.service, mc.change_type
        )
        SELECT * FROM msa_service_impact 
        WHERE iam_roles_affected > 0 OR assets_potentially_affected > 0
        ORDER BY (iam_roles_affected + assets_potentially_affected) DESC
    """)
    
    impact_chains = cursor.fetchall()
    
    for chain in impact_chains:
        output += f"### 📧 {chain['service']} {chain['change_type']}\n"
        output += f"```\n"
        output += f"MSA Change → {chain['iam_roles_affected']} IAM Roles → {chain['assets_potentially_affected']} Assets\n"
        output += f"```\n"
        
        # Calculate impact score
        impact_score = chain['iam_roles_affected'] + chain['assets_potentially_affected']
        impact_level = "🔴 HIGH" if impact_score > 50 else "🟠 MEDIUM" if impact_score > 10 else "🟡 LOW"
        output += f"**Impact Level:** {impact_level} (Score: {impact_score})\n\n"
    
    # 2. Security Finding → Asset Type → IAM Correlation
    output += "## 🛡️ Security Finding Ripple Effects\n\n"
    
    cursor.execute("""
        SELECT 
            sf.finding_class,
            sf.severity,
            COUNT(DISTINCT sf.resource_name) as affected_resources,
            COUNT(DISTINCT a.asset_type) as affected_asset_types,
            COUNT(DISTINCT ip.role) as related_iam_roles
        FROM security_findings sf
        LEFT JOIN assets a ON sf.resource_name = a.resource_name
        LEFT JOIN iam_policies ip ON sf.resource_name = ip.resource_name
        WHERE sf.state = 'ACTIVE'
        GROUP BY sf.finding_class, sf.severity
        HAVING affected_resources > 0
        ORDER BY affected_resources DESC, related_iam_roles DESC
        LIMIT 8
    """)
    
    finding_ripples = cursor.fetchall()
    
    for ripple in finding_ripples:
        severity_icon = {'CRITICAL': '🔴', 'HIGH': '🟠', 'MEDIUM': '🟡', 'LOW': '🟢'}.get(ripple['severity'], '⚪')
        output += f"{severity_icon} **{ripple['finding_class']}**\n"
        output += f"   • Resources Affected: {ripple['affected_resources']}\n"
        output += f"   • Asset Types Involved: {ripple['affected_asset_types']}\n"
        output += f"   • IAM Roles Connected: {ripple['related_iam_roles']}\n"
        
        # Show connection strength
        connection_strength = ripple['affected_asset_types'] + ripple['related_iam_roles']
        if connection_strength > 10:
            output += f"   • 🔗 **High interconnection** - Changes here affect multiple domains\n"
        elif connection_strength > 5:
            output += f"   • 🔗 Medium interconnection\n"
        else:
            output += f"   • 🔗 Low interconnection\n"
        output += "\n"
    
    # 3. Knowledge Base → Real-World Application Tracking
    output += "## 📚 Knowledge Base Application Tracking\n\n"
    
    cursor.execute("""
        SELECT 
            cs.standard_name,
            cs.severity,
            cs.language,
            COUNT(DISTINCT sf.resource_name) as violations_found
        FROM coding_standards cs
        LEFT JOIN security_findings sf ON (
            LOWER(sf.finding_class) LIKE '%' || LOWER(cs.standard_name) || '%' OR
            LOWER(sf.finding_class) LIKE '%secret%' AND LOWER(cs.standard_name) LIKE '%secret%' OR
            LOWER(sf.finding_class) LIKE '%test%' AND LOWER(cs.standard_name) LIKE '%test%'
        )
        GROUP BY cs.standard_name, cs.severity, cs.language
        ORDER BY violations_found DESC, 
                 CASE cs.severity WHEN 'ERROR' THEN 1 WHEN 'WARNING' THEN 2 ELSE 3 END
    """)
    
    kb_applications = cursor.fetchall()
    
    for app in kb_applications:
        if app['violations_found'] > 0:
            severity_icon = {'ERROR': '🔴', 'WARNING': '🟠', 'INFO': '🟡'}.get(app['severity'], '⚪')
            output += f"{severity_icon} **{app['standard_name']}** ({app['language']})\n"
            output += f"   • Potential violations detected: {app['violations_found']}\n"
            output += f"   • 🔄 **Feedback loop active** - Standard violations being tracked in real findings\n\n"
    
    # 4. Temporal Impact Analysis
    output += "## ⏰ Temporal Impact Patterns\n\n"
    
    output += "**Recent Change Velocity:**\n"
    
    cursor.execute("""
        SELECT 
            'MSA Changes' as change_type,
            COUNT(*) as recent_changes
        FROM msa_changes mc
        WHERE mc.effective_date >= date('now', '-30 days')
        
        UNION ALL
        
        SELECT 
            'New Security Findings' as change_type,
            COUNT(*) as recent_changes
        FROM security_findings sf
        WHERE sf.create_time >= datetime('now', '-30 days')
        
        UNION ALL
        
        SELECT 
            'Asset Changes' as change_type,
            COUNT(*) as recent_changes  
        FROM assets a
        WHERE a.create_time >= datetime('now', '-30 days')
    """)
    
    temporal_changes = cursor.fetchall()
    
    for change in temporal_changes:
        velocity_icon = "⚡" if change['recent_changes'] > 10 else "🟡" if change['recent_changes'] > 3 else "🟢"
        output += f"{velocity_icon} {change['change_type']}: {change['recent_changes']} in last 30 days\n"
    
    output += "\n## 🎯 Cross-Impact Optimization Recommendations\n\n"
    
    output += "**Strengthen Feedback Loops:**\n"
    output += "1. 🔄 Implement real-time MSA → IAM → Asset monitoring\n"
    output += "2. 📊 Create dashboard showing cross-domain impact metrics\n"
    output += "3. 🤖 Automate correlation between knowledge base violations and security findings\n"
    output += "4. 🔔 Set up alerts for high-impact security finding patterns\n"
    output += "5. 📈 Track remediation effectiveness across domains\n\n"
    
    output += "**Predictive Capabilities:**\n"
    output += "1. 🔮 Use historical patterns to predict MSA impact radius\n"
    output += "2. 🎯 Prioritize security findings based on cross-domain connections\n"
    output += "3. 📋 Pre-emptively update IAM policies before MSA effective dates\n"
    output += "4. 🛡️ Identify asset types most vulnerable to specific finding classes\n\n"
    
    return output