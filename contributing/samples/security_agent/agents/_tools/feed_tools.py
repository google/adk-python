"""
Tools for querying RSS feed data from BigQuery
Provides access to GCP release notes and security threat feeds
"""

from .base import (
    bq_client, check_client, PROJECT_ID,
    DEFAULT_DATASET, MAX_RESULTS,
    logger
)
from google.cloud import bigquery
import json


def query_gcp_release_notes(
    days_back: int = 30,
    security_only: bool = False,
    service_category: str = "",
    min_security_score: int = 0
) -> str:
    """
    Query Google Cloud Platform release notes from RSS feeds

    Args:
        days_back: Number of days back to search (default: 30)
        security_only: Only return security-related release notes (default: False)
        service_category: Filter by service category (compute, storage, security, etc.)
        min_security_score: Minimum security score (0-10, default: 0)

    Returns:
        Formatted string with release notes matching criteria
    """
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    # Build WHERE conditions
    where_conditions = []
    where_conditions.append(f"published_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)")

    if security_only:
        where_conditions.append("is_security_related = true")

    if service_category:
        where_conditions.append(f"service_category = '{service_category}'")

    if min_security_score > 0:
        where_conditions.append(f"security_score >= {min_security_score}")

    where_clause = " AND ".join(where_conditions)

    query = f"""
    SELECT
        title,
        description,
        link,
        source_feed,
        feed_name,
        published_date,
        service_category,
        security_keywords,
        security_score,
        is_security_related
    FROM `{PROJECT_ID}.{DEFAULT_DATASET}.gcp_release_notes`
    WHERE {where_clause}
    ORDER BY published_date DESC, security_score DESC
    LIMIT {MAX_RESULTS}
    """

    try:
        results = bq_client.query(query).result()

        # Convert to list for counting
        rows = list(results)

        if rows:
            output = [f"🆕 GCP Release Notes ({len(rows)} items, last {days_back} days)"]
            output.append("=" * 60)

            for row in rows:
                security_indicator = "🔒" if row.is_security_related else "📋"
                score_indicator = f"[Score: {row.security_score}]" if row.security_score > 0 else ""

                output.append(f"\n{security_indicator} {row.title} {score_indicator}")
                output.append(f"   Source: {row.feed_name}")
                output.append(f"   Category: {row.service_category}")
                output.append(f"   Published: {row.published_date}")

                if row.security_keywords:
                    keywords = ", ".join(row.security_keywords)
                    output.append(f"   Security Keywords: {keywords}")

                if row.description:
                    desc = row.description[:200] + "..." if len(row.description) > 200 else row.description
                    output.append(f"   Description: {desc}")

                output.append(f"   Link: {row.link}")
                output.append("-" * 40)

            return "\n".join(output)
        else:
            return f"No GCP release notes found for the last {days_back} days with the specified criteria."

    except Exception as e:
        return f"Error querying GCP release notes: {e}"


def query_security_threat_feeds(
    days_back: int = 7,
    severity: str = "",
    threat_type: str = "",
    min_cvss_score: float = 0.0,
    cloud_related_only: bool = False,
    immediate_action_only: bool = False
) -> str:
    """
    Query security threat feeds (CVE, advisories, threat intelligence)

    Args:
        days_back: Number of days back to search (default: 7)
        severity: Filter by severity (critical, high, medium, low)
        threat_type: Filter by threat type (vulnerability, malware, phishing, etc.)
        min_cvss_score: Minimum CVSS score (0.0-10.0, default: 0.0)
        cloud_related_only: Only return cloud-related threats (default: False)
        immediate_action_only: Only return threats requiring immediate action (default: False)

    Returns:
        Formatted string with security threats matching criteria
    """
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    # Build WHERE conditions
    where_conditions = []
    where_conditions.append(f"published_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)")

    if severity:
        where_conditions.append(f"severity = '{severity.lower()}'")

    if threat_type:
        where_conditions.append(f"threat_type = '{threat_type.lower()}'")

    if min_cvss_score > 0:
        where_conditions.append(f"cvss_score >= {min_cvss_score}")

    if cloud_related_only:
        where_conditions.append("is_cloud_related = true")

    if immediate_action_only:
        where_conditions.append("requires_immediate_action = true")

    where_clause = " AND ".join(where_conditions)

    query = f"""
    SELECT
        title,
        description,
        link,
        source_feed,
        feed_name,
        feed_type,
        published_date,
        cvss_score,
        cve_ids,
        cve_count,
        threat_type,
        severity,
        is_cloud_related,
        affected_products,
        requires_immediate_action
    FROM `{PROJECT_ID}.{DEFAULT_DATASET}.security_threat_feeds`
    WHERE {where_clause}
    ORDER BY
        CASE severity
            WHEN 'critical' THEN 1
            WHEN 'high' THEN 2
            WHEN 'medium' THEN 3
            WHEN 'low' THEN 4
            ELSE 5
        END,
        cvss_score DESC,
        published_date DESC
    LIMIT {MAX_RESULTS}
    """

    try:
        results = bq_client.query(query).result()

        # Convert to list for counting
        rows = list(results)

        if rows:
            output = [f"⚠️ Security Threat Feeds ({len(rows)} items, last {days_back} days)"]
            output.append("=" * 60)

            critical_count = sum(1 for row in rows if row.severity == 'critical')
            high_count = sum(1 for row in rows if row.severity == 'high')
            cve_count = sum(1 for row in rows if row.cve_count > 0)

            output.append(f"📊 Summary: {critical_count} Critical, {high_count} High, {cve_count} with CVEs")
            output.append("")

            for row in rows:
                # Severity indicators
                severity_icons = {
                    'critical': '🚨',
                    'high': '⚠️',
                    'medium': '⚡',
                    'low': 'ℹ️'
                }
                severity_icon = severity_icons.get(row.severity, '📋')

                cloud_indicator = "☁️" if row.is_cloud_related else ""
                action_indicator = "🎯" if row.requires_immediate_action else ""

                output.append(f"\n{severity_icon} {row.title} {cloud_indicator} {action_indicator}")
                output.append(f"   Source: {row.feed_name} ({row.feed_type})")
                output.append(f"   Severity: {row.severity.upper()}")
                output.append(f"   Threat Type: {row.threat_type}")
                output.append(f"   Published: {row.published_date}")

                if row.cvss_score > 0:
                    output.append(f"   CVSS Score: {row.cvss_score}/10.0")

                if row.cve_ids:
                    cve_list = ", ".join(row.cve_ids[:5])  # Show first 5 CVEs
                    if len(row.cve_ids) > 5:
                        cve_list += f" (+{len(row.cve_ids) - 5} more)"
                    output.append(f"   CVE IDs: {cve_list}")

                if row.affected_products:
                    products = ", ".join(row.affected_products[:3])
                    if len(row.affected_products) > 3:
                        products += f" (+{len(row.affected_products) - 3} more)"
                    output.append(f"   Affected Products: {products}")

                if row.description:
                    desc = row.description[:200] + "..." if len(row.description) > 200 else row.description
                    output.append(f"   Description: {desc}")

                output.append(f"   Link: {row.link}")
                output.append("-" * 40)

            return "\n".join(output)
        else:
            return f"No security threats found for the last {days_back} days with the specified criteria."

    except Exception as e:
        return f"Error querying security threat feeds: {e}"


def get_feed_statistics() -> str:
    """
    Get statistics about the RSS feed data

    Returns:
        Formatted string with feed statistics and freshness
    """
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    # Query for release notes statistics
    release_notes_query = f"""
    SELECT
        COUNT(*) as total_count,
        COUNT(CASE WHEN is_security_related THEN 1 END) as security_related_count,
        COUNT(CASE WHEN published_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY) THEN 1 END) as last_7_days,
        COUNT(CASE WHEN published_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY) THEN 1 END) as last_30_days,
        MAX(last_refreshed) as last_refreshed,
        source_feed,
        feed_name
    FROM `{PROJECT_ID}.{DEFAULT_DATASET}.gcp_release_notes`
    GROUP BY source_feed, feed_name
    ORDER BY total_count DESC
    """

    # Query for security feeds statistics
    security_feeds_query = f"""
    SELECT
        COUNT(*) as total_count,
        COUNT(CASE WHEN severity = 'critical' THEN 1 END) as critical_count,
        COUNT(CASE WHEN severity = 'high' THEN 1 END) as high_count,
        COUNT(CASE WHEN requires_immediate_action THEN 1 END) as action_required,
        COUNT(CASE WHEN is_cloud_related THEN 1 END) as cloud_related,
        COUNT(CASE WHEN published_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY) THEN 1 END) as last_7_days,
        MAX(last_refreshed) as last_refreshed,
        source_feed,
        feed_name,
        feed_type
    FROM `{PROJECT_ID}.{DEFAULT_DATASET}.security_threat_feeds`
    GROUP BY source_feed, feed_name, feed_type
    ORDER BY critical_count DESC, high_count DESC
    """

    try:
        output = ["📊 RSS Feed Statistics"]
        output.append("=" * 50)

        # Release Notes Statistics
        release_results = list(bq_client.query(release_notes_query).result())
        if release_results:
            output.append("\n🆕 GCP Release Notes Feeds:")
            output.append("-" * 30)

            total_releases = sum(row.total_count for row in release_results)
            total_security_releases = sum(row.security_related_count for row in release_results)
            total_recent = sum(row.last_7_days for row in release_results)

            output.append(f"Total Release Notes: {total_releases:,}")
            output.append(f"Security-Related: {total_security_releases:,}")
            output.append(f"Last 7 Days: {total_recent:,}")
            output.append("")

            for row in release_results:
                output.append(f"📋 {row.feed_name}")
                output.append(f"   Total: {row.total_count:,} | Security: {row.security_related_count:,} | Recent: {row.last_7_days:,}")
                output.append(f"   Last Updated: {row.last_refreshed}")

        # Security Feeds Statistics
        security_results = list(bq_client.query(security_feeds_query).result())
        if security_results:
            output.append("\n\n⚠️ Security Threat Feeds:")
            output.append("-" * 30)

            total_threats = sum(row.total_count for row in security_results)
            total_critical = sum(row.critical_count for row in security_results)
            total_high = sum(row.high_count for row in security_results)
            total_action_required = sum(row.action_required for row in security_results)
            total_cloud = sum(row.cloud_related for row in security_results)
            total_recent_threats = sum(row.last_7_days for row in security_results)

            output.append(f"Total Threats: {total_threats:,}")
            output.append(f"Critical: {total_critical:,} | High: {total_high:,}")
            output.append(f"Action Required: {total_action_required:,}")
            output.append(f"Cloud-Related: {total_cloud:,}")
            output.append(f"Last 7 Days: {total_recent_threats:,}")
            output.append("")

            for row in security_results:
                output.append(f"⚠️ {row.feed_name} ({row.feed_type})")
                output.append(f"   Total: {row.total_count:,} | Critical: {row.critical_count:,} | High: {row.high_count:,}")
                output.append(f"   Cloud: {row.cloud_related:,} | Action Required: {row.action_required:,}")
                output.append(f"   Last Updated: {row.last_refreshed}")

        return "\n".join(output)

    except Exception as e:
        return f"Error getting feed statistics: {e}"


def search_feeds_by_keyword(
    keyword: str,
    days_back: int = 30,
    include_release_notes: bool = True,
    include_security_feeds: bool = True
) -> str:
    """
    Search across all RSS feeds by keyword

    Args:
        keyword: Keyword to search for in titles and descriptions
        days_back: Number of days back to search (default: 30)
        include_release_notes: Include GCP release notes in search (default: True)
        include_security_feeds: Include security threat feeds in search (default: True)

    Returns:
        Formatted string with search results
    """
    try:
        check_client()
    except Exception as e:
        return f"Error: {e}"

    results = []

    if include_release_notes:
        # Search release notes
        release_query = f"""
        SELECT
            'release_note' as item_type,
            title,
            description,
            link,
            source_feed,
            feed_name,
            published_date,
            service_category,
            security_score,
            is_security_related
        FROM `{PROJECT_ID}.{DEFAULT_DATASET}.gcp_release_notes`
        WHERE (
            LOWER(title) LIKE '%{keyword.lower()}%' OR
            LOWER(description) LIKE '%{keyword.lower()}%'
        )
        AND published_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
        """

        try:
            release_results = list(bq_client.query(release_query).result())
            results.extend(release_results)
        except Exception as e:
            logger.warning(f"Error searching release notes: {e}")

    if include_security_feeds:
        # Search security feeds
        security_query = f"""
        SELECT
            'security_threat' as item_type,
            title,
            description,
            link,
            source_feed,
            feed_name,
            published_date,
            threat_type as service_category,
            CAST(cvss_score as INT64) as security_score,
            is_cloud_related as is_security_related
        FROM `{PROJECT_ID}.{DEFAULT_DATASET}.security_threat_feeds`
        WHERE (
            LOWER(title) LIKE '%{keyword.lower()}%' OR
            LOWER(description) LIKE '%{keyword.lower()}%'
        )
        AND published_date >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL {days_back} DAY)
        """

        try:
            security_results = list(bq_client.query(security_query).result())
            results.extend(security_results)
        except Exception as e:
            logger.warning(f"Error searching security feeds: {e}")

    if results:
        # Sort by published date, newest first
        results.sort(key=lambda x: x.published_date if x.published_date else datetime.min, reverse=True)

        output = [f"🔍 Search Results for '{keyword}' ({len(results)} items)"]
        output.append("=" * 60)

        for item in results[:MAX_RESULTS]:
            type_indicator = "🆕" if item.item_type == 'release_note' else "⚠️"
            security_indicator = "🔒" if item.is_security_related else ""

            output.append(f"\n{type_indicator} {item.title} {security_indicator}")
            output.append(f"   Source: {item.feed_name}")
            output.append(f"   Category: {item.service_category}")
            output.append(f"   Published: {item.published_date}")

            if item.description:
                desc = item.description[:200] + "..." if len(item.description) > 200 else item.description
                output.append(f"   Description: {desc}")

            output.append(f"   Link: {item.link}")
            output.append("-" * 40)

        return "\n".join(output)
    else:
        return f"No results found for keyword '{keyword}' in the last {days_back} days."