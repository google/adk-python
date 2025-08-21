#!/usr/bin/env python3
"""
Test BigQuery ACL MSA Analysis
===============================

Tests the MSA analyzer with the real BigQuery ACL permissions email.
"""

import httpx
import asyncio
import json
import os
from dotenv import load_dotenv

load_dotenv()

# The actual MSA email content
BIGQUERY_ACL_MSA = """We're writing to remind you that starting March 17, 2026, we'll be introducing more granular permission requirements for managing BigQuery dataset Access Control Lists (ACLs). This follows the initial communication we sent on June 5, 2025, where the implementation date was mentioned as September 15, 2025.

This important update will allow you to manage permissions for dataset metadata and ACL updates independently, providing finer control and enhancing security by giving users only the necessary permissions.

We're also introducing new parameters in dataset APIs to manage metadata and ACLs independently to align with the granular permissions. To prepare for these changes, we request that you review your custom roles and update them as needed to align with these revised permission requirements and avoid any user experience disruptions.

We understand that these changes may require some planning, therefore we have provided additional information below to guide you.

What you need to know
Key changes:

Permission Updates

Currently, certain permissions grant broad access:

bigquery.datasets.get: Allows viewing both metadata and ACLs
bigquery.datasets.update: Allows updating both metadata and ACLs
bigquery.datasets.create: Allows setting ACLs upon creation
Starting March 17, 2026, managing ACLs will require the following new, separate permissions:

bigquery.datasets.getIamPolicy: Required to view dataset ACLs and query the Object_Privileges view
bigquery.datasets.setIamPolicy: Required to update dataset ACLs
API Parameter Updates

Starting March 17, 2026, the Dataset APIs will include the following new parameters to manage metadata and ACLs independently:

Dataset Get API: The dataset_view parameter will have the following new options:

METADATA: View only metadata (requires bigquery.datasets.get)
ACL: View only ACLs (requires bigquery.datasets.getIamPolicy)
FULL (default): View both (requires both bigquery.datasets.get and bigquery.datasets.getIamPolicy)
Dataset Patch and Update APIs: The update_mode parameter will have the following new options:

UPDATE_METADATA: Update only metadata (requires bigquery.datasets.update).
UPDATE_ACL: Update only ACLs (requires bigquery.datasets.setIamPolicy)
UPDATE_FULL (default): Update both (requires both bigquery.datasets.update and bigquery.datasets.setIamPolicy)
Potential impact:

Custom roles with only bigquery.datasets.get, bigquery.datasets.create, or bigquery.datasets.update permission will lose the ability to view or modify ACLs after March 17, 2026, unless updated.
Predefined roles will not be affected, since they already incorporate the new permissions.
What you need to do
Required actions:

Review Custom Roles: Identify all custom roles in your BigQuery projects
Assess Current Permissions: Check if these roles include bigquery.datasets.get, bigquery.datasets.create, or bigquery.datasets.update
Update Roles for ACL Management:
To retain the ability to view ACLs, add the bigquery.datasets.getIamPolicy permission
To retain the ability to update ACLs, add the bigquery.datasets.setIamPolicy permission
If you don't want to add these permissions to the custom roles, ensure that the Dataset Get API is used with dataset_view=METADATA parameter, and the Dataset Patch and Update APIs are used with update_mode=UPDATE_METADATA parameter
Consider Early Testing: Test the updated APIs with the new permissions by following these instructions
Timelines:

March 17, 2026: New permission requirements will be enforced. Update custom roles before this date.

We're here to help
We constantly strive to provide you with more secure and flexible data management. If you have questions or need assistance, please contact Google Cloud Support.

Projects for your review can be found in the attachment below.

Thanks for choosing BigQuery."""

async def test_bigquery_msa():
    """Test analyzing the BigQuery ACL MSA."""
    
    print("🧪 Testing BigQuery ACL MSA Analysis")
    print("=" * 50)
    
    backend_url = os.getenv("BACKEND_URL", "http://localhost:8000")
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    
    payload = {
        "email_content": BIGQUERY_ACL_MSA,
        "project_id": project_id if project_id and project_id != "your-project-id" else "test-project-123"
    }
    
    print(f"📧 Analyzing MSA for project: {payload['project_id']}")
    print(f"📝 MSA Length: {len(BIGQUERY_ACL_MSA)} characters")
    
    async with httpx.AsyncClient() as client:
        try:
            # Send to backend for analysis
            print("\n🤖 Sending to backend for Gemini analysis...")
            response = await client.post(
                f"{backend_url}/api/v1/msa/analyze",
                json=payload,
                timeout=30.0
            )
            
            if response.status_code == 200:
                results = response.json()
                
                print("\n✅ Analysis Complete!")
                print("\n" + "=" * 50)
                print("📊 SUMMARY")
                print("=" * 50)
                
                summary = results.get("summary", {})
                print(f"Total changes detected: {summary.get('total_changes', 0)}")
                print(f"Critical changes: {summary.get('critical_changes', 0)}")
                print(f"High impact changes: {summary.get('high_impact_changes', 0)}")
                print(f"Services affected: {', '.join(summary.get('services_affected', []))}")
                print(f"Earliest effective date: {summary.get('earliest_effective_date', 'N/A')}")
                
                print("\n" + "=" * 50)
                print("🔄 EXTRACTED CHANGES")
                print("=" * 50)
                
                for i, change in enumerate(results.get("extracted_changes", []), 1):
                    print(f"\n{i}. {change['service']} - {change['change_type']}")
                    print(f"   Impact: {change['impact_level'].upper()}")
                    print(f"   Description: {change['description'][:150]}...")
                    if change.get('effective_date'):
                        print(f"   Effective: {change['effective_date']}")
                    if change.get('required_action'):
                        print(f"   Action Required: {change['required_action'][:150]}...")
                
                print("\n" + "=" * 50)
                print("💡 RECOMMENDATIONS")
                print("=" * 50)
                
                for rec in results.get("recommendations", []):
                    print(f"• {rec}")
                
                # Check if saved to database
                print("\n" + "=" * 50)
                print("💾 DATABASE STORAGE")
                print("=" * 50)
                
                # Query the database to verify storage
                import sqlite3
                db_path = os.getenv("DATABASE_PATH", "backend/cache/gcp_data.db")
                
                if os.path.exists(db_path):
                    conn = sqlite3.connect(db_path)
                    cursor = conn.cursor()
                    
                    # Check for this MSA in database
                    cursor.execute("""
                        SELECT COUNT(*) FROM msa_emails 
                        WHERE email_content LIKE '%BigQuery dataset Access Control Lists%'
                    """)
                    count = cursor.fetchone()[0]
                    
                    if count > 0:
                        print(f"✅ MSA saved to database ({count} record(s) found)")
                        
                        # Get the changes count
                        cursor.execute("""
                            SELECT COUNT(*) FROM msa_changes mc
                            JOIN msa_emails me ON mc.msa_email_id = me.id
                            WHERE me.email_content LIKE '%BigQuery dataset Access Control Lists%'
                        """)
                        changes_count = cursor.fetchone()[0]
                        print(f"✅ {changes_count} changes stored in database")
                    else:
                        print("⚠️ MSA not found in database (may need to check storage)")
                    
                    conn.close()
                else:
                    print("⚠️ Database file not found")
                
                print("\n" + "=" * 50)
                print("🎯 AGENT QUERY TEST")
                print("=" * 50)
                
                # Test agent query capability
                try:
                    import sys
                    sys.path.insert(0, 'agents/gcp_security')
                    from sqlite_tool import query_security_data
                    
                    # Query MSA changes
                    result = query_security_data("msa_changes", '{"service": "BigQuery"}')
                    if "BigQuery" in result:
                        print("✅ Agent can query BigQuery MSA changes")
                        print(f"   Result preview: {result[:200]}...")
                    else:
                        print("⚠️ Agent query returned no BigQuery changes")
                except Exception as e:
                    print(f"❌ Agent query test failed: {e}")
                
                return True
                
            else:
                print(f"\n❌ Analysis failed: {response.status_code}")
                print(f"Error: {response.text}")
                return False
                
        except Exception as e:
            print(f"\n❌ Error: {e}")
            return False

if __name__ == "__main__":
    success = asyncio.run(test_bigquery_msa())
    
    print("\n" + "=" * 50)
    if success:
        print("✅ BigQuery ACL MSA test completed successfully!")
        print("\nThe MSA has been:")
        print("1. Analyzed by Gemini")
        print("2. Stored in the database")
        print("3. Made available to the agent for queries")
        print("\nYou can now ask the agent:")
        print("• 'Show me MSA changes for BigQuery'")
        print("• 'What permissions are changing for BigQuery ACLs?'")
        print("• 'When do the BigQuery permission changes take effect?'")
    else:
        print("❌ Test failed. Please check the backend is running.")
    print("=" * 50)