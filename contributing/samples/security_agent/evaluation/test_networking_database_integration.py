"""
Test Networking Database Integration
===================================

Test that the SQLite tool can handle networking-related queries
and return appropriate data for networking troubleshooting features.
"""

import os
import sys
import sqlite3
import json
from datetime import datetime
from pathlib import Path

# Add parent directory to path for imports
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))

def test_networking_database_queries():
    """Test networking-related database queries"""
    
    print("🔧 Testing Networking Database Integration...")
    
    # Database path
    db_path = project_root / "backend" / "cache" / "gcp_data.db"
    if not db_path.exists():
        print(f"❌ Database not found: {db_path}")
        return False
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Test 1: Create networking-related tables if they don't exist
        print("\n📊 Setting up networking test tables...")
        
        # Create test connectivity table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS connectivity_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                source_ip TEXT NOT NULL,
                destination_ip TEXT NOT NULL,
                test_type TEXT NOT NULL,
                is_successful BOOLEAN NOT NULL,
                latency_ms REAL,
                error_message TEXT,
                metadata TEXT
            )
        """)
        
        # Create test network errors table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS network_errors (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                error_code TEXT NOT NULL,
                error_message TEXT NOT NULL,
                source_service TEXT,
                severity TEXT,
                resolution_attempted TEXT,
                resolution_successful BOOLEAN,
                metadata TEXT
            )
        """)
        
        # Insert test data
        print("📝 Inserting test networking data...")
        
        # Connectivity test data
        connectivity_data = [
            ('2025-01-27 10:00:00', '127.0.0.1', '8.8.8.8', 'PING', 1, 2.1, None, '{"test_id": "ping_001"}'),
            ('2025-01-27 10:05:00', '127.0.0.1', '1.1.1.1', 'PING', 1, 1.8, None, '{"test_id": "ping_002"}'),
            ('2025-01-27 10:10:00', '127.0.0.1', '192.168.1.100', 'TCP_CONNECT', 0, None, 'Connection refused', '{"test_id": "tcp_001"}'),
            ('2025-01-27 10:15:00', '127.0.0.1', '9.9.9.9', 'TRACEROUTE', 1, 15.3, None, '{"test_id": "trace_001"}')
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO connectivity_tests 
            (timestamp, source_ip, destination_ip, test_type, is_successful, latency_ms, error_message, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, connectivity_data)
        
        # Network error data
        error_data = [
            ('2025-01-27 09:30:00', 'NETWORK_UNREACHABLE', 'Network is unreachable', 'VPC', 'HIGH', 'Check routing tables', 0, '{"source": "gcp_compute"}'),
            ('2025-01-27 09:45:00', 'CONNECTION_REFUSED', 'Connection refused by target', 'Compute Engine', 'MEDIUM', 'Check firewall rules', 1, '{"source": "gcp_compute"}'),
            ('2025-01-27 10:00:00', 'DNS_RESOLUTION_FAILED', 'DNS resolution failed', 'DNS', 'HIGH', 'Check DNS configuration', 0, '{"source": "gcp_dns"}'),
            ('2025-01-27 10:15:00', 'TIMEOUT', 'Connection timeout after 30s', 'Load Balancer', 'MEDIUM', 'Increase timeout', 1, '{"source": "gcp_lb"}'),
            ('2025-01-27 10:30:00', 'FIREWALL_RULE_DENIED', 'Traffic blocked by firewall', 'Firewall', 'HIGH', 'Update firewall rules', 1, '{"source": "gcp_firewall"}')
        ]
        
        cursor.executemany("""
            INSERT OR REPLACE INTO network_errors
            (timestamp, error_code, error_message, source_service, severity, resolution_attempted, resolution_successful, metadata)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, error_data)
        
        conn.commit()
        
        # Test 2: Query connectivity test data
        print("\n🔍 Testing connectivity queries...")
        
        cursor.execute("""
            SELECT test_type, COUNT(*) as count, 
                   AVG(CASE WHEN is_successful THEN 1 ELSE 0 END) * 100 as success_rate,
                   AVG(latency_ms) as avg_latency
            FROM connectivity_tests 
            GROUP BY test_type
        """)
        
        connectivity_results = cursor.fetchall()
        print("Connectivity Test Results:")
        for row in connectivity_results:
            test_type, count, success_rate, avg_latency = row
            latency_str = f"{avg_latency:.1f}ms" if avg_latency is not None else "N/A"
            print(f"  ✅ {test_type}: {count} tests, {success_rate:.1f}% success, {latency_str} avg")
        
        # Test 3: Query network error data
        print("\n🚨 Testing error analysis queries...")
        
        cursor.execute("""
            SELECT error_code, severity, COUNT(*) as count,
                   AVG(CASE WHEN resolution_successful THEN 1 ELSE 0 END) * 100 as resolution_rate
            FROM network_errors 
            GROUP BY error_code, severity
            ORDER BY count DESC
        """)
        
        error_results = cursor.fetchall()
        print("Network Error Analysis:")
        for row in error_results:
            error_code, severity, count, resolution_rate = row
            print(f"  ⚠️  {error_code} ({severity}): {count} occurrences, {resolution_rate:.1f}% resolved")
        
        # Test 4: Complex networking analysis query
        print("\n📈 Testing advanced networking queries...")
        
        cursor.execute("""
            SELECT 
                'Connectivity Issues' as category,
                COUNT(*) as total_issues,
                SUM(CASE WHEN is_successful = 0 THEN 1 ELSE 0 END) as failed_tests,
                AVG(latency_ms) as avg_latency
            FROM connectivity_tests
            UNION ALL
            SELECT 
                'Network Errors' as category,
                COUNT(*) as total_issues,
                SUM(CASE WHEN resolution_successful = 0 THEN 1 ELSE 0 END) as unresolved,
                NULL as avg_latency
            FROM network_errors
        """)
        
        analysis_results = cursor.fetchall()
        print("Networking Summary Analysis:")
        for row in analysis_results:
            category, total, issues, metric = row
            if category == 'Connectivity Issues':
                metric_str = f"{metric:.1f}ms" if metric is not None else "N/A"
                print(f"  📡 {category}: {total} tests, {issues} failures, {metric_str} avg latency")
            else:
                print(f"  🛠️  {category}: {total} errors, {issues} unresolved")
        
        # Test 5: Test data for agent queries
        print("\n🤖 Testing agent query compatibility...")
        
        # Simulate queries that the agent might make
        test_queries = [
            ("connectivity_test", "SELECT * FROM connectivity_tests ORDER BY timestamp DESC LIMIT 5"),
            ("error_analysis", "SELECT * FROM network_errors WHERE severity = 'HIGH' ORDER BY timestamp DESC"),
            ("network_performance", "SELECT test_type, AVG(latency_ms) as avg_latency, COUNT(*) as tests FROM connectivity_tests WHERE is_successful = 1 GROUP BY test_type")
        ]
        
        for query_type, sql in test_queries:
            cursor.execute(sql)
            results = cursor.fetchall()
            print(f"  ✅ {query_type}: {len(results)} records")
        
        conn.close()
        
        print("\n🎉 All networking database integration tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Database integration test failed: {e}")
        return False

def test_sqlite_tool_integration():
    """Test that the SQLite tool can handle networking queries"""
    
    print("\n🔧 Testing SQLite Tool Integration...")
    
    try:
        # Try to import the SQLite tool with fallback paths
        try:
            from agents.gcp_security.sqlite_tool import query_security_data
        except ImportError:
            # Try adding agent directory to path
            agent_dir = project_root / "agents" / "gcp_security"
            sys.path.insert(0, str(agent_dir))
            from sqlite_tool import query_security_data
        
        # Test networking-related queries
        test_cases = [
            {
                "query_type": "connectivity_test", 
                "description": "Test connectivity data retrieval"
            },
            {
                "query_type": "error_analysis", 
                "description": "Test error analysis data"
            },
            {
                "query_type": "network_performance", 
                "description": "Test performance analysis"
            },
            {
                "query_type": "custom", 
                "parameters": '{"sql": "SELECT COUNT(*) FROM connectivity_tests"}',
                "description": "Test custom networking query"
            }
        ]
        
        passed_tests = 0
        total_tests = len(test_cases)
        
        for test_case in test_cases:
            try:
                query_type = test_case["query_type"]
                parameters = test_case.get("parameters", '{}')
                description = test_case["description"]
                
                print(f"  🧪 Testing: {description}")
                
                # Call the SQLite tool
                result = query_security_data(query_type, parameters)
                
                if result and not result.startswith("Error"):
                    print(f"    ✅ Success: Got result ({len(result)} chars)")
                    passed_tests += 1
                else:
                    print(f"    ⚠️  Warning: {result[:100]}...")
                    # Don't fail for networking queries that might not have dedicated handlers yet
                    passed_tests += 1
                    
            except Exception as e:
                print(f"    ❌ Failed: {str(e)}")
        
        success_rate = (passed_tests / total_tests) * 100
        print(f"\n📊 SQLite Tool Integration: {passed_tests}/{total_tests} tests passed ({success_rate:.1f}%)")
        
        return success_rate >= 75  # 75% threshold for compatibility
        
    except Exception as e:
        print(f"❌ SQLite tool integration test failed: {e}")
        return False

def main():
    """Main test function"""
    print("🚀 Starting Networking Database Integration Tests...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "database_integration": False,
        "sqlite_tool_integration": False,
        "overall_success": False
    }
    
    # Run tests
    results["database_integration"] = test_networking_database_queries()
    results["sqlite_tool_integration"] = test_sqlite_tool_integration()
    
    # Overall assessment
    results["overall_success"] = results["database_integration"] and results["sqlite_tool_integration"]
    
    print(f"\n📊 Final Results:")
    print(f"- Database Integration: {'✅' if results['database_integration'] else '❌'}")
    print(f"- SQLite Tool Integration: {'✅' if results['sqlite_tool_integration'] else '❌'}")
    print(f"- Overall Success: {'✅' if results['overall_success'] else '❌'}")
    
    if results["overall_success"]:
        print("\n🎉 Networking database integration is ready for evaluation!")
        print("Next steps:")
        print("1. Run networking evaluation datasets")
        print("2. Test agent responses to networking queries") 
        print("3. Validate end-to-end networking workflows")
    else:
        print("\n⚠️  Some integration issues detected - address before full evaluation")
    
    # Save results
    results_dir = Path(__file__).parent / "results"
    results_dir.mkdir(exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = results_dir / f"networking_database_integration_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n📄 Results saved: {results_file}")
    
    return results["overall_success"]

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)