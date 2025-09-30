#!/usr/bin/env python3
"""
Flask web application for the BigQuery Security Agent
Provides a web interface to interact with the ADK agent
"""

from flask import Flask, render_template, request, jsonify
from agents.agent import root_agent
from agents._tools.security_tools import (
    get_security_insights_summary,
    get_security_statistics
)
import json
import traceback
import re

app = Flask(__name__)

# Configure Flask
app.config['JSON_SORT_KEYS'] = False

@app.route('/')
def index():
    """Render the main page"""
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    """Handle chat requests"""
    try:
        data = request.get_json()
        message = data.get('message', '')

        if not message:
            return jsonify({
                'error': 'No message provided',
                'success': False
            }), 400

        # Call the ADK backend and format the response
        import requests
        import uuid

        # First create a session
        session_url = "http://localhost:8000/apps/agents/users/web-user/sessions"
        session_response = requests.post(session_url)

        if session_response.status_code == 200:
            session_data = session_response.json()
            session_id = session_data.get('id', str(uuid.uuid4()))
        else:
            session_id = str(uuid.uuid4())

        # Now send the message
        run_url = "http://localhost:8000/run"
        payload = {
            "appName": "agents",
            "userId": "web-user",
            "sessionId": session_id,
            "newMessage": {
                "parts": [{"text": message}],
                "role": "user"
            }
        }

        try:
            adk_response = requests.post(run_url, json=payload)

            if adk_response.status_code == 200:
                result = adk_response.json()

                # Extract and format the response
                response = ""
                if isinstance(result, list):
                    for event in result:
                        if isinstance(event, dict) and "content" in event:
                            content = event["content"]
                            if isinstance(content, dict) and "parts" in content:
                                for part in content["parts"]:
                                    if isinstance(part, dict) and "text" in part:
                                        text = part["text"]
                                        # Format the response better
                                        response += text

                if not response:
                    response = "No response from agent. Please try again."
            else:
                response = f"Error from ADK backend: Status {adk_response.status_code}"

        except requests.exceptions.ConnectionError:
            response = "Cannot connect to ADK backend. Make sure 'adk web' is running on port 8000."
        except Exception as e:
            response = f"Error: {str(e)}"

        return jsonify({
            'response': response,
            'success': True
        })
    except Exception as e:
        print(f"Error in chat endpoint: {e}")
        print(traceback.format_exc())
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        # Test that we can access the agent
        _ = root_agent.name
        return jsonify({
            'status': 'healthy',
            'agent': root_agent.name,
            'model': root_agent.model
        })
    except Exception as e:
        return jsonify({
            'status': 'unhealthy',
            'error': str(e)
        }), 500

@app.route('/agent-info')
def agent_info():
    """Get information about the agent"""
    try:
        tools_info = []
        for tool in root_agent.tools:
            # Handle FunctionTool objects
            if hasattr(tool, 'function'):
                tools_info.append({
                    'name': tool.function.__name__,
                    'description': tool.function.__doc__ or 'No description'
                })
            else:
                tools_info.append({
                    'name': str(tool),
                    'description': 'Tool information not available'
                })

        return jsonify({
            'name': root_agent.name,
            'model': root_agent.model,
            'tools': tools_info,
            'instruction_preview': root_agent.instruction[:200] + '...' if len(root_agent.instruction) > 200 else root_agent.instruction
        })
    except Exception as e:
        return jsonify({
            'error': str(e),
            'success': False
        }), 500

@app.route('/api/metrics')
def get_metrics():
    """Get security metrics data for dashboard"""
    try:
        # Get summary data
        summary = get_security_insights_summary()

        # Parse summary to extract metrics
        metrics = {}
        lines = summary.split('\n')
        for line in lines:
            if 'Total Records:' in line:
                metrics['total_records'] = int(re.search(r'([\d,]+)', line).group(1).replace(',', ''))
            elif 'Unique Categories:' in line:
                metrics['categories'] = int(re.search(r'(\d+)', line).group(1))
            elif 'Severity Levels:' in line:
                metrics['severity_levels'] = int(re.search(r'(\d+)', line).group(1))
            elif 'Resource Types:' in line:
                metrics['resource_types'] = int(re.search(r'(\d+)', line).group(1))
            elif 'Table Size:' in line:
                match = re.search(r'([\d,]+) rows', line)
                if match:
                    metrics['table_rows'] = int(match.group(1).replace(',', ''))

        return jsonify(metrics)
    except Exception as e:
        print(f"Error getting metrics: {e}")
        # Return mock data for development
        return jsonify({
            'total_records': 1247,
            'categories': 8,
            'severity_levels': 4,
            'resource_types': 12,
            'table_rows': 1247
        })

@app.route('/api/severity-distribution')
def get_severity_distribution():
    """Get severity distribution data for charts"""
    try:
        # Get statistics grouped by severity
        stats = get_security_statistics('severity')

        # Parse the response to extract data
        distribution = []
        lines = stats.split('\n')
        for line in lines:
            # Look for lines with severity data
            if 'HIGH' in line or 'CRITICAL' in line or 'MEDIUM' in line or 'LOW' in line:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    severity = parts[0].strip()
                    # Extract count from the line
                    match = re.search(r'(\d+)', parts[1])
                    if match:
                        count = int(match.group(1))
                        distribution.append({
                            'severity': severity,
                            'count': count
                        })

        if not distribution:
            # Return mock data for development
            distribution = [
                {'severity': 'CRITICAL', 'count': 45},
                {'severity': 'HIGH', 'count': 234},
                {'severity': 'MEDIUM', 'count': 567},
                {'severity': 'LOW', 'count': 401}
            ]

        return jsonify(distribution)
    except Exception as e:
        print(f"Error getting severity distribution: {e}")
        # Return mock data for development
        return jsonify([
            {'severity': 'CRITICAL', 'count': 45},
            {'severity': 'HIGH', 'count': 234},
            {'severity': 'MEDIUM', 'count': 567},
            {'severity': 'LOW', 'count': 401}
        ])

@app.route('/api/category-distribution')
def get_category_distribution():
    """Get category distribution data for charts"""
    try:
        # Get statistics grouped by category
        stats = get_security_statistics('category')

        # Parse the response to extract data
        distribution = []
        lines = stats.split('\n')
        for line in lines:
            # Look for lines with category data (after the header)
            if ':' in line and 'Category' not in line and 'Total' not in line:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    category = parts[0].strip()
                    # Extract count from the line
                    match = re.search(r'(\d+)', parts[1])
                    if match:
                        count = int(match.group(1))
                        distribution.append({
                            'category': category,
                            'count': count
                        })

        if not distribution:
            # Return mock data for development
            distribution = [
                {'category': 'IAM_POLICY', 'count': 312},
                {'category': 'FIREWALL_RULES', 'count': 189},
                {'category': 'DATA_EXPOSURE', 'count': 267},
                {'category': 'COMPLIANCE', 'count': 145},
                {'category': 'NETWORK_SECURITY', 'count': 334}
            ]

        return jsonify(distribution)
    except Exception as e:
        print(f"Error getting category distribution: {e}")
        # Return mock data for development
        return jsonify([
            {'category': 'IAM_POLICY', 'count': 312},
            {'category': 'FIREWALL_RULES', 'count': 189},
            {'category': 'DATA_EXPOSURE', 'count': 267},
            {'category': 'COMPLIANCE', 'count': 145},
            {'category': 'NETWORK_SECURITY', 'count': 334}
        ])

@app.route('/api/resource-type-distribution')
def get_resource_type_distribution():
    """Get resource type distribution data for charts"""
    try:
        # Get statistics grouped by resource_type
        stats = get_security_statistics('resource_type')

        # Parse the response to extract data
        distribution = []
        lines = stats.split('\n')
        for line in lines:
            # Look for lines with resource type data
            if ':' in line and 'Resource' not in line and 'Total' not in line:
                parts = line.strip().split(':')
                if len(parts) >= 2:
                    resource_type = parts[0].strip()
                    # Extract count from the line
                    match = re.search(r'(\d+)', parts[1])
                    if match:
                        count = int(match.group(1))
                        distribution.append({
                            'resource_type': resource_type,
                            'count': count
                        })

        if not distribution:
            # Return mock data for development
            distribution = [
                {'resource_type': 'compute.instances', 'count': 234},
                {'resource_type': 'storage.buckets', 'count': 156},
                {'resource_type': 'iam.serviceAccounts', 'count': 289},
                {'resource_type': 'container.clusters', 'count': 78},
                {'resource_type': 'compute.networks', 'count': 123},
                {'resource_type': 'bigquery.datasets', 'count': 367}
            ]

        return jsonify(distribution)
    except Exception as e:
        print(f"Error getting resource type distribution: {e}")
        # Return mock data for development
        return jsonify([
            {'resource_type': 'compute.instances', 'count': 234},
            {'resource_type': 'storage.buckets', 'count': 156},
            {'resource_type': 'iam.serviceAccounts', 'count': 289},
            {'resource_type': 'container.clusters', 'count': 78},
            {'resource_type': 'compute.networks', 'count': 123},
            {'resource_type': 'bigquery.datasets', 'count': 367}
        ])

if __name__ == '__main__':
    print("🚀 Starting Flask app for BigQuery Security Agent")
    print(f"   Agent: {root_agent.name}")
    print(f"   Model: {root_agent.model}")
    print(f"   Tools: {len(root_agent.tools)} tools available")
    print("\n📍 Server running at: http://localhost:5000")
    print("   Health check: http://localhost:5000/health")
    print("   Agent info: http://localhost:5000/agent-info")

    app.run(debug=True, port=5000)