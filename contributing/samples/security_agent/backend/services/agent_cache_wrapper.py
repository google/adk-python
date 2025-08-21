"""
Agent Cache Wrapper - Cache-first tools for the security agent.

This module provides cache-first implementations of agent tools that:
1. Check SQLite cache first for fast responses
2. Fall back to API calls only when cache is empty or stale
3. Update cache with fresh data when needed
4. Handle errors gracefully with cached fallbacks
"""

import logging
import os
import asyncio
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

# Import cache components
try:
    from .data_fetcher import DataFetcher
    from .cache_manager import get_cache_manager
    CACHE_AVAILABLE = True
except ImportError:
    try:
        from data_fetcher import DataFetcher
        from cache_manager import get_cache_manager
        CACHE_AVAILABLE = True
    except ImportError:
        CACHE_AVAILABLE = False

logger = logging.getLogger(__name__)
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')


class AgentCacheWrapper:
    """Cache-first wrapper for agent tools."""
    
    def __init__(self, project_id: str = PROJECT_ID):
        self.project_id = project_id
        if CACHE_AVAILABLE:
            self.data_fetcher = DataFetcher(project_id)
            self.cache_manager = get_cache_manager()
        else:
            self.data_fetcher = None
            self.cache_manager = None
    
    def analyze_storage_sync(self, force_refresh: bool = False) -> str:
        """
        Sync version of storage analysis using cached data.
        
        Args:
            force_refresh: Force fresh data fetch
            
        Returns:
            Formatted storage analysis results
        """
        if not CACHE_AVAILABLE:
            return "⚠️ Cache not available - storage analysis requires cached data"
        
        try:
            # Query cached storage buckets directly (sync)
            buckets = self.data_fetcher.query_storage_buckets()
            
            if not buckets:
                return "⚠️ No storage data found in cache. Try refreshing data first."
            
            logger.info(f"Found {len(buckets)} buckets in cache")
            logger.debug(f"First bucket sample: {buckets[0] if buckets else 'None'}")
            
            # Analyze the cached bucket data - ensure we handle different data formats
            total_buckets = len(buckets)
            public_buckets = []
            unencrypted_buckets = []
            
            for i, bucket in enumerate(buckets):
                try:
                    # The bucket should be a dict from query_storage_buckets()
                    if not isinstance(bucket, dict):
                        logger.error(f"Bucket {i} is not a dict: {type(bucket)}")
                        continue
                    
                    bucket_data = bucket.copy()  # Work with a copy
                    
                    # Parse the 'data' field if it exists and is a JSON string
                    if 'data' in bucket_data and isinstance(bucket_data['data'], str):
                        try:
                            import json
                            parsed_data = json.loads(bucket_data['data'])
                            # Merge parsed data with the existing bucket_data, preferring explicit columns
                            for key, value in parsed_data.items():
                                if key not in bucket_data or bucket_data[key] is None:
                                    bucket_data[key] = value
                        except json.JSONDecodeError as json_error:
                            logger.warning(f"Could not parse JSON data for bucket {i}: {json_error}")
                    
                    # Check public access - handle various field formats
                    is_public = False
                    public_access = bucket_data.get('public_access')
                    if public_access:
                        if isinstance(public_access, str):
                            is_public = public_access.lower() in ['true', 'public', '1', 'yes']
                        elif isinstance(public_access, bool):
                            is_public = public_access
                    
                    if is_public:
                        public_buckets.append(bucket_data)
                    
                    # Check encryption - handle various field formats  
                    is_unencrypted = False
                    encryption = bucket_data.get('encryption')
                    if encryption:
                        if isinstance(encryption, str):
                            try:
                                import json
                                encryption_data = json.loads(encryption)
                                is_unencrypted = not encryption_data.get('enabled', True)
                            except:
                                # If not JSON, assume it's a simple string indicator
                                is_unencrypted = encryption.lower() in ['false', 'disabled', '0', 'no']
                        elif isinstance(encryption, dict):
                            is_unencrypted = not encryption.get('enabled', True)
                        elif isinstance(encryption, bool):
                            is_unencrypted = not encryption
                    else:
                        # No encryption field means unencrypted
                        is_unencrypted = True
                    
                    if is_unencrypted:
                        unencrypted_buckets.append(bucket_data)
                        
                except Exception as bucket_error:
                    logger.error(f"Error processing bucket {i}: {bucket_error}, bucket type: {type(bucket)}, bucket: {bucket}")
                    continue
            
            # Generate security analysis
            analysis = f"""📊 Storage Security Analysis (from cache):

🗄️ **Total Buckets**: {total_buckets}
🌐 **Public Buckets**: {len(public_buckets)} {'⚠️' if public_buckets else '✅'}
🔒 **Unencrypted Buckets**: {len(unencrypted_buckets)} {'⚠️' if unencrypted_buckets else '✅'}

"""
            
            # Add bucket details
            if public_buckets:
                analysis += "⚠️ **Public Buckets Found**:\n"
                for bucket in public_buckets[:3]:  # Show first 3
                    bucket_name = bucket.get('name', 'Unknown')
                    bucket_location = bucket.get('location', 'Unknown')
                    analysis += f"  • {bucket_name} (Location: {bucket_location})\n"
                if len(public_buckets) > 3:
                    analysis += f"  • ...and {len(public_buckets) - 3} more\n"
                analysis += "\n"
            
            if unencrypted_buckets:
                analysis += "🔓 **Unencrypted Buckets Found**:\n"
                for bucket in unencrypted_buckets[:3]:
                    bucket_name = bucket.get('name', 'Unknown')
                    analysis += f"  • {bucket_name}\n"
                if len(unencrypted_buckets) > 3:
                    analysis += f"  • ...and {len(unencrypted_buckets) - 3} more\n"
                analysis += "\n"
            
            # Security recommendations
            if public_buckets or unencrypted_buckets:
                analysis += "🎯 **Recommendations**:\n"
                if public_buckets:
                    analysis += "  • Review and restrict public access to storage buckets\n"
                if unencrypted_buckets:
                    analysis += "  • Enable encryption for all storage buckets\n"
            else:
                analysis += "✅ **All storage buckets follow security best practices!**\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sync storage analysis: {e}")
            return f"❌ Error analyzing cached storage data: {str(e)}"
    
    def discover_assets_sync(self, force_refresh: bool = False) -> str:
        """
        Sync version of asset discovery using cached data.
        
        Args:
            force_refresh: Force fresh data fetch (ignored in sync version)
            
        Returns:
            Formatted asset discovery results
        """
        if not CACHE_AVAILABLE:
            return "⚠️ Cache not available - asset discovery requires cached data"
        
        try:
            # Get summary stats from cache
            stats = self.data_fetcher.get_summary_stats()
            
            if not stats or all(v == 0 for k, v in stats.items() if k != 'last_fetch'):
                return "⚠️ No asset data found in cache. Try refreshing data first."
            
            # Generate asset summary from cached data
            total_assets = sum(v for k, v in stats.items() if k != 'last_fetch' and isinstance(v, int))
            
            analysis = f"""📊 GCP Asset Discovery (from cache):

🏗️ **Total Resources**: {total_assets:,}
🖥️ **Compute Instances**: {stats.get('compute_instances', 0)}
🗄️ **Storage Buckets**: {stats.get('storage_buckets', 0)}
🌐 **Networks**: {stats.get('networks', 0)}
🔒 **IAM Accounts**: {stats.get('iam_accounts', 0)}
🔍 **Security Findings**: {stats.get('security_findings', 0)}
🔐 **Secrets**: {stats.get('secrets', 0)}

"""
            
            # Add freshness info
            if stats.get('last_fetch'):
                analysis += f"📅 **Last Updated**: {stats['last_fetch']}\n"
            
            if total_assets > 0:
                analysis += "\n✅ **Asset data is available in cache for detailed analysis**"
            else:
                analysis += "\n⚠️ **No assets found - try triggering a data refresh**"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sync asset discovery: {e}")
            return f"❌ Error discovering cached assets: {str(e)}"
    
    def analyze_security_sync(self, force_refresh: bool = False) -> str:
        """
        Sync version of security analysis using cached data.
        
        Args:
            force_refresh: Force fresh data fetch (ignored in sync version)
            
        Returns:
            Formatted security analysis results
        """
        if not CACHE_AVAILABLE:
            return "⚠️ Cache not available - security analysis requires cached data"
        
        try:
            # Query cached security findings directly
            findings = self.data_fetcher.query_security_findings()
            
            if not findings:
                return "⚠️ No security findings found in cache. Try refreshing data first."
            
            # Categorize findings by severity with proper data handling
            critical = []
            high = []
            medium = []
            low = []
            
            for finding in findings:
                try:
                    if not isinstance(finding, dict):
                        logger.warning(f"Finding is not a dict: {type(finding)}")
                        continue
                        
                    finding_data = finding.copy()
                    
                    # Parse the 'data' field if it exists and is a JSON string
                    if 'data' in finding_data and isinstance(finding_data['data'], str):
                        try:
                            import json
                            parsed_data = json.loads(finding_data['data'])
                            # Merge parsed data with the existing finding_data
                            for key, value in parsed_data.items():
                                if key not in finding_data or finding_data[key] is None:
                                    finding_data[key] = value
                        except json.JSONDecodeError:
                            logger.warning(f"Could not parse JSON data for finding")
                    
                    severity = finding_data.get('severity', '').upper()
                    if severity == 'CRITICAL':
                        critical.append(finding_data)
                    elif severity == 'HIGH':
                        high.append(finding_data)
                    elif severity == 'MEDIUM':
                        medium.append(finding_data)
                    elif severity == 'LOW':
                        low.append(finding_data)
                        
                except Exception as finding_error:
                    logger.error(f"Error processing finding: {finding_error}")
                    continue
            
            total_findings = len(findings)
            
            # Generate security analysis
            analysis = f"""🛡️ Security Analysis (from cache):

🚨 **Total Findings**: {total_findings}
🔴 **Critical**: {len(critical)}
🟠 **High**: {len(high)}
🟡 **Medium**: {len(medium)}
🔵 **Low**: {len(low)}

"""
            
            # Risk assessment
            risk_score = (len(critical) * 10 + len(high) * 5 + len(medium) * 2 + len(low) * 1)
            if risk_score == 0:
                risk_level = "✅ **EXCELLENT**"
            elif risk_score <= 10:
                risk_level = "🟢 **LOW**"
            elif risk_score <= 25:
                risk_level = "🟡 **MEDIUM**"
            elif risk_score <= 50:
                risk_level = "🟠 **HIGH**"
            else:
                risk_level = "🔴 **CRITICAL**"
            
            analysis += f"📊 **Risk Level**: {risk_level} (Score: {risk_score})\n\n"
            
            # Show top critical/high findings
            priority_findings = critical + high
            if priority_findings:
                analysis += "🎯 **Priority Issues**:\n"
                for finding in priority_findings[:3]:  # Show top 3
                    severity = finding.get('severity', 'UNKNOWN')
                    category = finding.get('category', 'Unknown')
                    resource = finding.get('resource_name', 'Unknown')
                    analysis += f"  • **{severity}**: {category} on `{resource}`\n"
                
                if len(priority_findings) > 3:
                    analysis += f"  • ...and {len(priority_findings) - 3} more priority issues\n"
            else:
                analysis += "✅ **No critical or high severity issues found!**\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error in sync security analysis: {e}")
            return f"❌ Error analyzing cached security data: {str(e)}"
    
    async def discover_assets_cached(self, force_refresh: bool = False) -> str:
        """
        Discover GCP assets using cache-first approach.
        
        Args:
            force_refresh: Force API fetch instead of using cache
        
        Returns:
            Formatted asset discovery results
        """
        if not CACHE_AVAILABLE:
            return "⚠️ Cache not available - run 'refresh data' to populate cache first"
        
        try:
            # Check cache first unless force refresh
            if not force_refresh:
                # Query the main assets table which has all resources
                import sqlite3
                from pathlib import Path
                
                db_path = Path(__file__).parent.parent / "cache" / "gcp_data.db"
                if db_path.exists():
                    with sqlite3.connect(str(db_path)) as conn:
                        cursor = conn.cursor()
                        cursor.execute("SELECT COUNT(*) FROM assets")
                        asset_count = cursor.fetchone()[0]
                        
                        if asset_count > 0:
                            # Get assets by type
                            cursor.execute("""
                                SELECT asset_type, COUNT(*) as count 
                                FROM assets 
                                GROUP BY asset_type 
                                ORDER BY count DESC
                            """)
                            asset_types = cursor.fetchall()
                            
                            return self._format_cached_assets_from_main_table(asset_count, asset_types)
                
                # Fallback to specific tables
                instances = self.data_fetcher.query_compute_instances()
                buckets = self.data_fetcher.query_storage_buckets()
                
                if instances or buckets:
                    return self._format_cached_assets(instances, buckets)
            
            # If cache empty or force refresh, trigger background refresh
            if force_refresh:
                await self._trigger_background_refresh()
                # Return current cache while refresh runs in background
                instances = self.data_fetcher.query_compute_instances()
                buckets = self.data_fetcher.query_storage_buckets()
                result = self._format_cached_assets(instances, buckets)
                result += "\n\n🔄 *Background data refresh initiated for latest data*"
                return result
            
            # Cache is empty, suggest refresh
            return (
                "📊 **No cached asset data found**\n\n"
                "To get asset discovery results:\n"
                "1. Run 'refresh data' to populate cache (takes ~30-60 seconds)\n"
                "2. Or ask me to 'discover assets with refresh' to force fresh data\n\n"
                "💡 *Cache-first approach eliminates timeout errors and provides instant responses*"
            )
            
        except Exception as e:
            logger.error(f"Error in cached asset discovery: {e}")
            return f"❌ Error accessing cached data: {str(e)}"
    
    async def analyze_security_cached(self, force_refresh: bool = False) -> str:
        """
        Analyze security findings using cache-first approach.
        
        Args:
            force_refresh: Force fresh data fetch
            
        Returns:
            Formatted security analysis results  
        """
        if not CACHE_AVAILABLE:
            return "⚠️ Cache not available - security analysis requires cached data"
        
        try:
            # Check cache first
            if not force_refresh:
                findings = self.data_fetcher.query_security_findings()
                
                if findings:
                    return self._format_cached_security_findings(findings)
            
            # If cache empty or force refresh needed
            if force_refresh:
                await self._trigger_background_refresh()
                findings = self.data_fetcher.query_security_findings()
                result = self._format_cached_security_findings(findings)
                result += "\n\n🔄 *Background data refresh initiated for latest findings*"
                return result
            
            # Cache is empty - provide sample findings 
            return self._get_sample_security_findings()
            
        except Exception as e:
            logger.error(f"Error in cached security analysis: {e}")
            return f"❌ Error accessing cached security data: {str(e)}"
    
    async def analyze_iam_cached(self, force_refresh: bool = False) -> str:
        """
        Analyze IAM using cache-first approach.
        
        Args:
            force_refresh: Force fresh data fetch
            
        Returns:
            Formatted IAM analysis results
        """
        if not CACHE_AVAILABLE:
            return "⚠️ Cache not available - IAM analysis requires cached data"
        
        try:
            # Check cache first
            if not force_refresh:
                # Query cached IAM data
                import sqlite3
                from pathlib import Path
                
                db_path = Path(__file__).parent.parent / "cache" / "gcp_data.db"
                if db_path.exists():
                    with sqlite3.connect(str(db_path)) as conn:
                        conn.row_factory = sqlite3.Row
                        cursor = conn.execute(
                            "SELECT * FROM iam_accounts WHERE project_id = ?",
                            [self.project_id]
                        )
                        accounts = [dict(row) for row in cursor]
                        
                        if accounts:
                            return self._format_cached_iam_analysis(accounts)
            
            # If cache empty, provide helpful message
            return (
                "🔐 **IAM Analysis - Cache Empty**\n\n"
                "To get IAM analysis:\n"
                "1. Run 'refresh data' to fetch and cache IAM data\n"
                "2. Or ask for 'IAM analysis with refresh' to force fresh data\n\n"
                "💡 *Cached IAM analysis includes service accounts, keys, and permissions*"
            )
            
        except Exception as e:
            logger.error(f"Error in cached IAM analysis: {e}")
            return f"❌ Error accessing cached IAM data: {str(e)}"
    
    async def analyze_storage_cached(self, force_refresh: bool = False) -> str:
        """
        Analyze storage using cache-first approach.
        
        Args:
            force_refresh: Force fresh data fetch
            
        Returns:
            Formatted storage analysis results
        """
        if not CACHE_AVAILABLE:
            return "⚠️ Cache not available - storage analysis requires cached data"
        
        try:
            # Check cache first
            if not force_refresh:
                buckets = self.data_fetcher.query_storage_buckets()
                
                if buckets:
                    return self._format_cached_storage_analysis(buckets)
            
            # If cache empty
            return (
                "🗄️ **Storage Analysis - Cache Empty**\n\n"
                "To get storage security analysis:\n"
                "1. Run 'refresh data' to fetch and cache storage data\n"
                "2. Or ask for 'storage analysis with refresh' to force fresh data\n\n"
                "💡 *Cached storage analysis includes bucket permissions, encryption, and public access*"
            )
            
        except Exception as e:
            logger.error(f"Error in cached storage analysis: {e}")
            return f"❌ Error accessing cached storage data: {str(e)}"
    
    async def get_cache_stats(self) -> str:
        """Get cache statistics and status."""
        if not CACHE_AVAILABLE:
            return "⚠️ Cache not available"
        
        try:
            stats = self.data_fetcher.get_summary_stats()
            
            output = "📊 **Cache Statistics**\n\n"
            output += f"Project: {self.project_id}\n\n"
            
            output += "**Cached Resources:**\n"
            output += f"• Compute Instances: {stats.get('compute_instances', 0)}\n"
            output += f"• Storage Buckets: {stats.get('storage_buckets', 0)}\n"
            output += f"• Networks: {stats.get('networks', 0)}\n"
            output += f"• Firewall Rules: {stats.get('firewall_rules', 0)}\n"
            output += f"• IAM Accounts: {stats.get('iam_accounts', 0)}\n"
            output += f"• Databases: {stats.get('databases', 0)}\n"
            output += f"• Security Findings: {stats.get('security_findings', 0)}\n\n"
            
            last_fetch = stats.get('last_fetch')
            if last_fetch:
                output += f"**Last Updated:** {last_fetch}\n"
            else:
                output += "**Status:** Cache empty - run 'refresh data' to populate\n"
            
            output += "\n💡 *Cache provides instant responses and eliminates timeout errors*"
            
            return output
            
        except Exception as e:
            logger.error(f"Error getting cache stats: {e}")
            return f"❌ Error accessing cache statistics: {str(e)}"
    
    async def _trigger_background_refresh(self):
        """Trigger background data refresh."""
        try:
            import httpx
            async with httpx.AsyncClient(timeout=5.0) as client:
                response = await client.post(f"http://localhost:8000/api/v1/data/warmup/{self.project_id}")
                if response.status_code != 200:
                    logger.warning(f"Background refresh failed: {response.status_code}")
        except Exception as e:
            logger.warning(f"Could not trigger background refresh: {e}")
    
    def _format_cached_assets(self, instances, buckets) -> str:
        """Format cached asset data."""
        output = f"🔍 **Asset Discovery Results** (from cache)\n\n"
        output += f"**Project:** {self.project_id}\n"
        output += f"**Total Assets:** {len(instances) + len(buckets)}\n\n"
        
        if instances:
            output += f"**Compute Instances ({len(instances)}):**\n"
            for instance in instances[:5]:
                status = instance.get('status', 'unknown')
                zone = instance.get('zone', 'unknown')
                status_emoji = "🟢" if status == "RUNNING" else "🔴" if status == "TERMINATED" else "🟡"
                output += f"  {status_emoji} {instance['name']} ({zone}) - {status}\n"
            
            if len(instances) > 5:
                output += f"    ... and {len(instances) - 5} more instances\n"
            output += "\n"
        
        if buckets:
            output += f"**Storage Buckets ({len(buckets)}):**\n"
            for bucket in buckets[:5]:
                location = bucket.get('location', 'unknown')
                public_access = bucket.get('public_access', 'private')
                access_emoji = "🔴" if public_access == "public" else "🟢"
                output += f"  {access_emoji} {bucket['name']} ({location}) - {public_access}\n"
            
            if len(buckets) > 5:
                output += f"    ... and {len(buckets) - 5} more buckets\n"
            output += "\n"
        
        if not instances and not buckets:
            output += "⚠️ No assets found in cache\n\n"
        
        output += "⚡ *Results from local cache - very fast!*\n"
        output += "💡 Run 'refresh data' to update cache with latest data"
        
        return output
    
    def _format_cached_assets_from_main_table(self, asset_count, asset_types) -> str:
        """Format cached asset data from main assets table."""
        output = f"🔍 **Asset Discovery Results** (from cache)\n\n"
        output += f"**Project:** {self.project_id}\n"
        output += f"**Total Assets:** {asset_count}\n\n"
        
        if asset_types:
            output += "**Assets by Type:**\n"
            for asset_type, count in asset_types[:10]:  # Show top 10 types
                # Simplify asset type names for readability
                type_name = asset_type.split('/')[-1] if '/' in asset_type else asset_type
                type_name = type_name.replace('googleapis.com', '').replace('.', ' ').title()
                
                # Add appropriate emoji
                emoji = "💻" if "instance" in type_name.lower() else \
                        "🗄️" if "bucket" in type_name.lower() or "storage" in type_name.lower() else \
                        "🌐" if "network" in type_name.lower() else \
                        "🔒" if "firewall" in type_name.lower() or "security" in type_name.lower() else \
                        "🔑" if "key" in type_name.lower() or "secret" in type_name.lower() else \
                        "🏷️" if "label" in type_name.lower() else \
                        "📊"
                
                output += f"  {emoji} {type_name}: {count}\n"
            
            if len(asset_types) > 10:
                remaining = sum(count for _, count in asset_types[10:])
                output += f"  📋 Other types: {remaining}\n"
        
        output += "\n⚡ *Results from local cache - very fast!*\n"
        output += "💡 Run 'refresh data' to update cache with latest data"
        
        return output
    
    def _format_cached_security_findings(self, findings) -> str:
        """Format cached security findings."""
        if not findings:
            return self._get_sample_security_findings()
        
        # Count by severity
        severity_counts = {"CRITICAL": 0, "HIGH": 0, "MEDIUM": 0, "LOW": 0}
        for finding in findings:
            severity = finding.get('severity', 'UNKNOWN')
            if severity in severity_counts:
                severity_counts[severity] += 1
        
        output = f"🛡️ **Security Analysis Results** (from cache)\n\n"
        output += f"**Project:** {self.project_id}\n"
        output += f"**Total Findings:** {len(findings)}\n\n"
        
        output += "**By Severity:**\n"
        for severity, count in severity_counts.items():
            if count > 0:
                emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(severity, "⚪")
                output += f"  {emoji} {severity}: {count}\n"
        output += "\n"
        
        # Show top findings
        output += "**Top Findings:**\n"
        for finding in findings[:5]:
            severity = finding.get('severity', 'UNKNOWN')
            category = finding.get('category', 'UNKNOWN')
            description = finding.get('description', 'No description')[:80]
            emoji = {"CRITICAL": "🔴", "HIGH": "🟠", "MEDIUM": "🟡", "LOW": "🔵"}.get(severity, "⚪")
            output += f"  {emoji} **{severity}** - {category}: {description}...\n"
        
        if len(findings) > 5:
            output += f"    ... and {len(findings) - 5} more findings\n"
        
        output += "\n⚡ *Results from local cache - very fast!*\n"
        output += "💡 Run 'refresh data' to update findings"
        
        return output
    
    def _format_cached_iam_analysis(self, accounts) -> str:
        """Format cached IAM analysis."""
        output = f"🔐 **IAM Analysis Results** (from cache)\n\n"
        output += f"**Project:** {self.project_id}\n"
        output += f"**Service Accounts:** {len(accounts)}\n\n"
        
        # Analyze accounts
        disabled_count = len([a for a in accounts if a.get('disabled')])
        keys_count = sum(len(eval(a.get('keys', '[]'))) for a in accounts)
        
        output += "**Key Metrics:**\n"
        output += f"• Active Accounts: {len(accounts) - disabled_count}\n"
        output += f"• Disabled Accounts: {disabled_count}\n"
        output += f"• Total Keys: {keys_count}\n\n"
        
        # Show sample accounts
        output += "**Service Accounts:**\n"
        for account in accounts[:5]:
            status = "🔴 Disabled" if account.get('disabled') else "🟢 Active"
            keys = eval(account.get('keys', '[]'))
            output += f"  {status} {account.get('email', 'unknown')}\n"
            output += f"    Keys: {len(keys)}\n"
        
        if len(accounts) > 5:
            output += f"    ... and {len(accounts) - 5} more accounts\n"
        
        output += "\n⚡ *Results from local cache - very fast!*"
        
        return output
    
    def _format_cached_storage_analysis(self, buckets) -> str:
        """Format cached storage analysis."""
        # Count by access level
        public_buckets = [b for b in buckets if b.get('public_access') == 'public']
        encrypted_buckets = [b for b in buckets if b.get('encryption') != 'Google-managed']
        
        output = f"🗄️ **Storage Analysis Results** (from cache)\n\n"
        output += f"**Project:** {self.project_id}\n"
        output += f"**Total Buckets:** {len(buckets)}\n\n"
        
        output += "**Security Status:**\n"
        output += f"• Public Buckets: {len(public_buckets)} 🔴\n"
        output += f"• Private Buckets: {len(buckets) - len(public_buckets)} 🟢\n"
        output += f"• Customer-Encrypted: {len(encrypted_buckets)}\n\n"
        
        if public_buckets:
            output += "**🚨 Public Buckets (High Risk):**\n"
            for bucket in public_buckets[:5]:
                output += f"  🔴 {bucket['name']} ({bucket.get('location', 'unknown')})\n"
            
            if len(public_buckets) > 5:
                output += f"    ... and {len(public_buckets) - 5} more public buckets\n"
            output += "\n"
        
        # Show sample buckets
        output += "**Storage Buckets:**\n"
        for bucket in buckets[:5]:
            access_emoji = "🔴" if bucket.get('public_access') == 'public' else "🟢"
            location = bucket.get('location', 'unknown')
            storage_class = bucket.get('storage_class', 'unknown')
            output += f"  {access_emoji} {bucket['name']} ({location}) - {storage_class}\n"
        
        if len(buckets) > 5:
            output += f"    ... and {len(buckets) - 5} more buckets\n"
        
        output += "\n⚡ *Results from local cache - very fast!*"
        
        return output
    
    def _get_sample_security_findings(self) -> str:
        """Get sample security findings when cache is empty."""
        return (
            f"🛡️ **Security Analysis** (sample data)\n\n"
            f"**Project:** {self.project_id}\n\n"
            "**Sample Security Findings:**\n"
            "🔴 **CRITICAL** - Public Bucket: Storage bucket publicly accessible\n"
            "🟠 **HIGH** - Weak Credentials: Service account key >90 days old\n"
            "🟡 **MEDIUM** - Firewall Misconfiguration: Overly permissive rules\n\n"
            "💡 **To get real findings:**\n"
            "1. Run 'refresh data' to populate security findings cache\n"
            "2. Or ask for 'security analysis with refresh' for fresh data\n\n"
            "⚡ *Cache-first approach provides instant responses*"
        )


# Global instance
_cache_wrapper = None

def get_agent_cache_wrapper() -> AgentCacheWrapper:
    """Get or create the agent cache wrapper instance."""
    global _cache_wrapper
    if _cache_wrapper is None:
        _cache_wrapper = AgentCacheWrapper()
    return _cache_wrapper