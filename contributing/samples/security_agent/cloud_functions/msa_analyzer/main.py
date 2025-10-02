"""
Cloud Function for MSA (Multi-Service Analyzer)
Monitors GCP release notes and analyzes impacts
Designed to be triggered by Cloud Scheduler (e.g., daily)
"""

import os
import json
import logging
from datetime import datetime
from flask import Request, jsonify

# Import MSA analyzer
import sys
sys.path.append('/workspace')
from msa_analyzer import MSAAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def analyze_releases(request: Request):
    """
    HTTP Cloud Function entry point for MSA analysis.

    Trigger with Cloud Scheduler:
    - Schedule: "0 9 * * *" (daily at 9 AM)
    - Target Type: HTTP
    - URL: https://REGION-PROJECT.cloudfunctions.net/msa-analyzer
    - HTTP Method: POST
    - Body: {"days_back": 7}
    """

    try:
        # Parse request
        request_json = request.get_json(silent=True)

        # Get parameters with defaults
        days_back = 7
        if request_json:
            days_back = request_json.get('days_back', 7)

        logger.info(f"Starting MSA analysis for last {days_back} days")

        # Initialize analyzer
        project_id = os.environ.get('GCP_PROJECT', os.environ.get('GOOGLE_CLOUD_PROJECT'))
        analyzer = MSAAnalyzer(project_id=project_id)

        # Run analysis
        report = analyzer.analyze_release_notes(days_back)

        # Log summary
        logger.info(f"Analysis complete: {report['summary']}")

        # Check for critical issues
        if report['summary']['critical_issues'] > 0:
            logger.warning(f"⚠️ {report['summary']['critical_issues']} critical issues found!")

            # Here you could trigger additional alerts
            # e.g., send to Pub/Sub, create incident, etc.
            _send_critical_alerts(report)

        # Store results in BigQuery (already done in analyzer)

        # Return summary response
        response = {
            'success': True,
            'analysis_id': report['analysis_id'],
            'timestamp': report['timestamp'],
            'summary': {
                'total_changes': report['summary']['total_changes_analyzed'],
                'services_affected': report['summary']['active_services_affected'],
                'risk_level': report['summary']['overall_risk_level'],
                'critical_issues': report['summary']['critical_issues']
            },
            'top_recommendations': report['recommendations'][:3] if report['recommendations'] else [],
            'message': _generate_summary_message(report)
        }

        return jsonify(response), 200

    except Exception as e:
        logger.error(f"Error in MSA analysis: {str(e)}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

def _generate_summary_message(report: dict) -> str:
    """Generate human-readable summary message"""

    risk_emoji = {
        'low': '🟢',
        'medium': '🟡',
        'high': '🔴'
    }.get(report['summary']['overall_risk_level'], '⚪')

    message_parts = [
        f"MSA analyzed {report['summary']['total_changes_analyzed']} changes",
        f"affecting {report['summary']['active_services_affected']} services."
    ]

    message_parts.append(f"Overall risk: {risk_emoji} {report['summary']['overall_risk_level'].upper()}")

    if report['summary']['critical_issues'] > 0:
        message_parts.append(f"⚠️ {report['summary']['critical_issues']} CRITICAL issues require immediate attention!")

    # Add impact summary
    impacts = []
    if report['security_impact']['risk_level'] != 'low':
        impacts.append(f"Security: {report['security_impact']['risk_level']}")
    if report['billing_impact']['estimated_impact'] != 'neutral':
        impacts.append(f"Billing: {report['billing_impact']['estimated_impact']}")
    if report['compliance_impact']['impact_level'] != 'low':
        impacts.append(f"Compliance: {report['compliance_impact']['impact_level']}")

    if impacts:
        message_parts.append(f"Impacts: {', '.join(impacts)}")

    return ' '.join(message_parts)

def _send_critical_alerts(report: dict):
    """Send alerts for critical issues"""

    try:
        # Example: Send to Pub/Sub for further processing
        from google.cloud import pubsub_v1

        publisher = pubsub_v1.PublisherClient()
        project_id = os.environ.get('GCP_PROJECT', os.environ.get('GOOGLE_CLOUD_PROJECT'))
        topic_name = 'msa-critical-alerts'
        topic_path = publisher.topic_path(project_id, topic_name)

        for rec in report['recommendations']:
            if rec['priority'] == 'critical':
                message = {
                    'analysis_id': report['analysis_id'],
                    'timestamp': report['timestamp'],
                    'priority': 'critical',
                    'category': rec['category'],
                    'action': rec['action'],
                    'deadline': rec['deadline'],
                    'details': rec.get('details', ''),
                    'link': rec.get('link', '')
                }

                # Publish message
                future = publisher.publish(
                    topic_path,
                    json.dumps(message).encode('utf-8'),
                    priority='critical',
                    category=rec['category']
                )

                logger.info(f"Alert published: {future.result()}")

    except Exception as e:
        logger.error(f"Failed to send alerts: {e}")
        # Don't fail the function if alerting fails

# For local testing
if __name__ == "__main__":
    from flask import Flask, request
    app = Flask(__name__)

    @app.route('/', methods=['POST'])
    def local_test():
        return analyze_releases(request)

    app.run(host='0.0.0.0', port=8080)