#!/usr/bin/env python3
"""
Lightweight Flask app for Cloud Run deployment
Serves as API gateway to Cloud Functions
"""

import os
import json
import logging
from flask import Flask, request, jsonify
import requests
from google.auth import default
from google.auth.transport.requests import Request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get configuration from environment
PROJECT_ID = os.getenv('GOOGLE_CLOUD_PROJECT', 'your-project-id')
REGION = os.getenv('GOOGLE_CLOUD_REGION', 'us-central1')

# Cloud Function endpoints (configure these)
FUNCTIONS = {
    'fetch_firewall_rules': f'https://{REGION}-{PROJECT_ID}.cloudfunctions.net/fetch-firewall-rules',
    'fetch_service_account_roles': f'https://{REGION}-{PROJECT_ID}.cloudfunctions.net/fetch-service-account-roles',
    'analyze_security': f'https://{REGION}-{PROJECT_ID}.cloudfunctions.net/analyze-security'
}

def get_auth_token():
    """Get authentication token for calling Cloud Functions"""
    try:
        credentials, project = default()
        auth_req = Request()
        credentials.refresh(auth_req)
        return credentials.token
    except Exception as e:
        logger.error(f"Failed to get auth token: {e}")
        return None

@app.route('/health')
def health():
    """Health check endpoint for Cloud Run"""
    return jsonify({
        'status': 'healthy',
        'service': 'security-agent-api',
        'project': PROJECT_ID
    })

@app.route('/')
def index():
    """Root endpoint"""
    return jsonify({
        'service': 'GCP Security Agent API',
        'version': '1.0.0',
        'endpoints': {
            '/health': 'Health check',
            '/api/firewall/rules': 'Get firewall rules',
            '/api/iam/service-accounts': 'Get service account roles',
            '/api/security/analyze': 'Analyze security posture'
        }
    })

@app.route('/api/firewall/rules', methods=['GET'])
def get_firewall_rules():
    """Proxy to firewall rules Cloud Function"""
    try:
        token = get_auth_token()
        headers = {'Authorization': f'Bearer {token}'} if token else {}

        response = requests.get(
            FUNCTIONS['fetch_firewall_rules'],
            headers=headers,
            timeout=30
        )

        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout'}), 504
    except Exception as e:
        logger.error(f"Error calling firewall function: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/iam/service-accounts', methods=['GET'])
def get_service_accounts():
    """Proxy to service account roles Cloud Function"""
    try:
        token = get_auth_token()
        headers = {'Authorization': f'Bearer {token}'} if token else {}

        response = requests.get(
            FUNCTIONS['fetch_service_account_roles'],
            headers=headers,
            timeout=30
        )

        return jsonify(response.json()), response.status_code
    except requests.exceptions.Timeout:
        return jsonify({'error': 'Request timeout'}), 504
    except Exception as e:
        logger.error(f"Error calling IAM function: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/security/analyze', methods=['POST'])
def analyze_security():
    """Analyze security posture"""
    try:
        data = request.get_json()
        resource_type = data.get('resource_type', 'all')

        # Aggregate data from multiple functions
        results = {
            'timestamp': os.environ.get('K_REVISION', 'local'),
            'project': PROJECT_ID,
            'analysis': {}
        }

        # Get firewall rules if needed
        if resource_type in ['all', 'network']:
            token = get_auth_token()
            headers = {'Authorization': f'Bearer {token}'} if token else {}

            firewall_response = requests.get(
                FUNCTIONS['fetch_firewall_rules'],
                headers=headers,
                timeout=30
            )
            if firewall_response.status_code == 200:
                results['analysis']['firewall'] = firewall_response.json()

        # Get IAM data if needed
        if resource_type in ['all', 'iam']:
            token = get_auth_token()
            headers = {'Authorization': f'Bearer {token}'} if token else {}

            iam_response = requests.get(
                FUNCTIONS['fetch_service_account_roles'],
                headers=headers,
                timeout=30
            )
            if iam_response.status_code == 200:
                results['analysis']['iam'] = iam_response.json()

        return jsonify(results)

    except Exception as e:
        logger.error(f"Error in security analysis: {e}")
        return jsonify({'error': str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    """Handle 404 errors"""
    return jsonify({'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    logger.error(f"Internal server error: {error}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False)