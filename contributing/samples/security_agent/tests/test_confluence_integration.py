"""
Test script for Confluence integration with ADK Security Agent

Tests both the Confluence tools and the BigQuery sync functionality.
"""

import os
import sys
import json
from pathlib import Path
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import the tools
from agents._tools.confluence_tools import (
    search_confluence_documentation,
    get_confluence_document,
    analyze_confluence_coverage,
    get_confluence_statistics,
    refresh_confluence_cache
)


def test_confluence_search():
    """Test searching Confluence documentation."""
    print("\n🔍 Testing Confluence search...")

    # Test basic search
    result = search_confluence_documentation(
        query="security policies",
        limit=5
    )

    print(f"Search result: {json.dumps(result, indent=2)}")

    if result.get("success"):
        print(f"✅ Found {result.get('count', 0)} documents")
        for doc in result.get("results", [])[:3]:
            print(f"  - {doc.get('title')} (Space: {doc.get('space_key')})")
    else:
        print(f"❌ Search failed: {result.get('error')}")

    return result


def test_confluence_statistics():
    """Test getting Confluence cache statistics."""
    print("\n📊 Testing Confluence statistics...")

    result = get_confluence_statistics()

    if result.get("success"):
        stats = result.get("cache_statistics", {})
        print(f"✅ Cache Statistics:")
        print(f"  - Total documents: {stats.get('total_documents', 0)}")
        print(f"  - Unique spaces: {stats.get('unique_spaces', 0)}")
        print(f"  - Cache status: {stats.get('cache_status', 'unknown')}")
        print(f"  - Cache age: {stats.get('cache_age_hours', 'N/A')} hours")

        if stats.get("space_breakdown"):
            print("\n  Space breakdown:")
            for space, count in stats["space_breakdown"].items():
                print(f"    - {space}: {count} documents")
    else:
        print(f"❌ Failed to get statistics: {result.get('error')}")

    return result


def test_coverage_analysis():
    """Test documentation coverage analysis."""
    print("\n📈 Testing coverage analysis...")

    topics = [
        "IAM security",
        "Network security",
        "Data encryption",
        "Compliance policies",
        "Incident response"
    ]

    result = analyze_confluence_coverage(topics)

    if result.get("success"):
        print(f"✅ Coverage Analysis:")
        print(f"  - Topics analyzed: {result.get('topics_analyzed', 0)}")
        print(f"  - Topics documented: {result.get('topics_documented', 0)}")
        print(f"  - Coverage percentage: {result.get('coverage_percentage', 0)}%")

        print("\n  Details by topic:")
        for topic, coverage in result.get("coverage_details", {}).items():
            status = "✅" if coverage["documented"] else "❌"
            print(f"    {status} {topic}: {coverage['document_count']} documents")

        print("\n  Recommendations:")
        for rec in result.get("recommendations", []):
            print(f"    - {rec}")
    else:
        print(f"❌ Coverage analysis failed: {result.get('error')}")

    return result


def test_document_retrieval():
    """Test retrieving a specific document."""
    print("\n📄 Testing document retrieval...")

    # First search for a document
    search_result = search_confluence_documentation("security", limit=1)

    if search_result.get("success") and search_result.get("results"):
        doc_id = search_result["results"][0].get("id")

        if doc_id:
            print(f"  Retrieving document ID: {doc_id}")
            doc_result = get_confluence_document(doc_id, include_content=False)

            if doc_result.get("success"):
                doc = doc_result.get("document", {})
                print(f"✅ Retrieved document:")
                print(f"  - Title: {doc.get('title')}")
                print(f"  - Space: {doc.get('space_key')}")
                print(f"  - Modified: {doc.get('modified_date')}")
                print(f"  - URL: {doc.get('url')}")
            else:
                print(f"❌ Failed to retrieve document: {doc_result.get('error')}")
        else:
            print("❌ No document ID found in search results")
    else:
        print("❌ No documents found to test retrieval")


def test_cache_refresh():
    """Test refreshing the Confluence cache."""
    print("\n♻️ Testing cache refresh...")

    # Check if we should refresh
    stats = get_confluence_statistics()
    if stats.get("cache_statistics", {}).get("cache_status") == "fresh":
        print("ℹ️ Cache is fresh, skipping refresh test")
        return

    result = refresh_confluence_cache(spaces=["SEC"], force=False)

    if result.get("success"):
        print(f"✅ Cache refresh completed:")
        print(f"  - Documents fetched: {result.get('documents_fetched', 0)}")
        print(f"  - Spaces refreshed: {result.get('spaces_refreshed', [])}")
        print(f"  - Message: {result.get('message')}")
    else:
        print(f"❌ Cache refresh failed: {result.get('error')}")
        print(f"  Message: {result.get('message')}")


def test_bigquery_integration():
    """Test BigQuery integration (requires Cloud Function deployed)."""
    print("\n☁️ Testing BigQuery integration...")

    try:
        from google.cloud import bigquery

        # Initialize BigQuery client
        client = bigquery.Client()

        # Check if table exists
        table_id = f"{os.getenv('GOOGLE_CLOUD_PROJECT', 'mgm-digitalconcierge')}.security_data.confluence_documents"

        try:
            table = client.get_table(table_id)
            print(f"✅ BigQuery table exists: {table_id}")

            # Get row count
            query = f"SELECT COUNT(*) as count FROM `{table_id}`"
            result = client.query(query).result()
            for row in result:
                print(f"  - Document count: {row.count}")

            # Get sample data
            query = f"""
                SELECT title, space_key, document_type, modified_date
                FROM `{table_id}`
                ORDER BY modified_date DESC
                LIMIT 5
            """
            result = client.query(query).result()

            print("\n  Recent documents:")
            for row in result:
                print(f"    - {row.title} ({row.space_key}) - {row.document_type}")

        except Exception as e:
            print(f"❌ BigQuery table not found or error: {str(e)}")
            print("  Run the Cloud Function deployment script to create the table")

    except ImportError:
        print("⚠️ google-cloud-bigquery not installed, skipping BigQuery test")
    except Exception as e:
        print(f"❌ BigQuery test failed: {str(e)}")


def main():
    """Run all tests."""
    print("=" * 60)
    print("🧪 CONFLUENCE INTEGRATION TEST SUITE")
    print("=" * 60)

    # Check environment setup
    print("\n🔧 Environment Configuration:")
    print(f"  - Confluence URL: {os.getenv('CONFLUENCE_URL', 'Not configured')}")
    print(f"  - Confluence Spaces: {os.getenv('CONFLUENCE_SPACES', 'SEC,POLICY,GCP')}")
    print(f"  - Cache DB: {os.getenv('CONFLUENCE_CACHE_DB', 'backend/cache/confluence_cache.db')}")
    print(f"  - GCP Project: {os.getenv('GOOGLE_CLOUD_PROJECT', 'Not configured')}")

    # Run tests
    test_confluence_statistics()
    test_confluence_search()
    test_coverage_analysis()
    test_document_retrieval()
    test_cache_refresh()
    test_bigquery_integration()

    print("\n" + "=" * 60)
    print("✅ TEST SUITE COMPLETED")
    print("=" * 60)


if __name__ == "__main__":
    main()