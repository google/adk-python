"""
PDF Report Generator for Service Evaluations
============================================

Generates professional PDF reports for GCP service security evaluations.
Uses reportlab for PDF generation with fallback to simple HTML->PDF if needed.
"""

import io
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional
import base64
import logging

logger = logging.getLogger(__name__)

# Try to import reportlab, fallback to simple solution if not available
REPORTLAB_AVAILABLE = False  # Disable for now, use HTML-based approach
logger.info("Using HTML-based PDF generation approach")


class PDFReportGenerator:
    """Generate PDF reports for service evaluations."""
    
    def __init__(self):
        self.styles = getSampleStyleSheet() if REPORTLAB_AVAILABLE else None
        if REPORTLAB_AVAILABLE:
            self._setup_custom_styles()
    
    def _setup_custom_styles(self):
        """Setup custom styles for the PDF."""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Title'],
            fontSize=24,
            textColor=colors.HexColor('#1a73e8'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))
        
        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#34495e'),
            spaceAfter=12,
            spaceBefore=12
        ))
        
        # Risk level styles
        self.styles.add(ParagraphStyle(
            name='RiskHigh',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.red,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskMedium',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.orange,
            fontName='Helvetica-Bold'
        ))
        
        self.styles.add(ParagraphStyle(
            name='RiskLow',
            parent=self.styles['Normal'],
            fontSize=12,
            textColor=colors.green,
            fontName='Helvetica-Bold'
        ))
    
    def generate_evaluation_pdf(self, evaluation_data: Dict[str, Any]) -> bytes:
        """
        Generate a PDF report for a service evaluation.
        
        Args:
            evaluation_data: Service evaluation data from the API
            
        Returns:
            PDF file as bytes
        """
        if not REPORTLAB_AVAILABLE:
            return self._generate_simple_pdf(evaluation_data)
        
        # Create PDF in memory
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=72,
            leftMargin=72,
            topMargin=72,
            bottomMargin=18,
        )
        
        # Container for the 'Flowable' objects
        elements = []
        
        # Add logo/header
        elements.append(Paragraph(
            "🔐 GCP Security Evaluation Report",
            self.styles['CustomTitle']
        ))
        
        # Add timestamp
        elements.append(Paragraph(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            self.styles['Normal']
        ))
        elements.append(Spacer(1, 20))
        
        # Service information section
        elements.append(Paragraph("Service Information", self.styles['CustomSubtitle']))
        
        service_data = [
            ['Service Name:', evaluation_data.get('service_name', 'N/A')],
            ['Description:', evaluation_data.get('description', 'N/A')],
            ['Release Stage:', evaluation_data.get('release_stage', 'N/A')],
            ['Status:', 'Enabled' if evaluation_data.get('is_enabled') else 'Not Enabled']
        ]
        
        service_table = Table(service_data, colWidths=[2*inch, 4*inch])
        service_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.black),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
            ('GRID', (0, 0), (-1, -1), 1, colors.grey)
        ]))
        elements.append(service_table)
        elements.append(Spacer(1, 20))
        
        # Use cases
        if evaluation_data.get('use_cases'):
            elements.append(Paragraph("Use Cases", self.styles['CustomSubtitle']))
            for use_case in evaluation_data['use_cases']:
                elements.append(Paragraph(f"• {use_case}", self.styles['Normal']))
            elements.append(Spacer(1, 20))
        
        # Security Assessment
        if evaluation_data.get('security_assessment'):
            assessment = evaluation_data['security_assessment']
            
            elements.append(Paragraph("Security Assessment", self.styles['CustomSubtitle']))
            
            # Overall risk score
            risk_score = assessment.get('risk_score', 0)
            risk_style = self._get_risk_style(risk_score)
            elements.append(Paragraph(
                f"Overall Risk Score: {risk_score}/10",
                self.styles[risk_style]
            ))
            elements.append(Spacer(1, 10))
            
            # Risk profile breakdown
            if assessment.get('risk_profile'):
                elements.append(Paragraph("Risk Profile Breakdown", self.styles['Heading3']))
                risk_profile = assessment['risk_profile']
                
                risk_data = [
                    ['Risk Category', 'Score (0-10)'],
                    ['Data Exposure', str(risk_profile.get('data_exposure', 0))],
                    ['Misconfiguration', str(risk_profile.get('misconfiguration', 0))],
                    ['Attack Surface', str(risk_profile.get('attack_surface', 0))],
                    ['Compliance Violation', str(risk_profile.get('compliance_violation', 0))]
                ]
                
                risk_table = Table(risk_data, colWidths=[3*inch, 2*inch])
                risk_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#34495e')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 10),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                    ('GRID', (0, 0), (-1, -1), 1, colors.grey)
                ]))
                elements.append(risk_table)
                elements.append(Spacer(1, 20))
            
            # Security details
            elements.append(Paragraph("Security Configuration", self.styles['Heading3']))
            
            security_details = [
                ['Network Exposure:', assessment.get('network_exposure', 'N/A')],
                ['Data Encryption:', assessment.get('data_encryption', 'N/A')],
                ['Data Residency:', assessment.get('data_residency', 'N/A')]
            ]
            
            details_table = Table(security_details, colWidths=[2*inch, 4*inch])
            details_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.lightgrey),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 10),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
                ('GRID', (0, 0), (-1, -1), 1, colors.grey)
            ]))
            elements.append(details_table)
            elements.append(Spacer(1, 20))
            
            # Compliance certifications
            if assessment.get('compliance_certifications'):
                elements.append(Paragraph("Compliance Certifications", self.styles['Heading3']))
                for cert in assessment['compliance_certifications']:
                    elements.append(Paragraph(f"✓ {cert}", self.styles['Normal']))
                elements.append(Spacer(1, 20))
            
            # IAM Permissions
            if assessment.get('iam_permissions'):
                elements.append(Paragraph("Required IAM Permissions", self.styles['Heading3']))
                permissions = assessment['iam_permissions'][:10]  # First 10
                for perm in permissions:
                    elements.append(Paragraph(f"• {perm}", self.styles['Code']))
                if len(assessment['iam_permissions']) > 10:
                    elements.append(Paragraph(
                        f"... and {len(assessment['iam_permissions']) - 10} more",
                        self.styles['Italic']
                    ))
                elements.append(Spacer(1, 20))
            
            # Threat model
            if assessment.get('threat_model_summary'):
                elements.append(Paragraph("Threat Model Summary", self.styles['Heading3']))
                elements.append(Paragraph(
                    assessment['threat_model_summary'],
                    self.styles['Normal']
                ))
                elements.append(Spacer(1, 20))
        
        # Recommendations section
        elements.append(PageBreak())
        elements.append(Paragraph("Recommendations", self.styles['CustomSubtitle']))
        
        recommendations = self._generate_recommendations(evaluation_data)
        for i, rec in enumerate(recommendations, 1):
            elements.append(Paragraph(f"{i}. {rec}", self.styles['Normal']))
            elements.append(Spacer(1, 10))
        
        # Footer
        elements.append(Spacer(1, 30))
        elements.append(Paragraph(
            "This report was generated automatically by the GCP Security Agent",
            self.styles['Italic']
        ))
        
        # Build PDF
        doc.build(elements)
        
        # Get PDF bytes
        pdf_bytes = buffer.getvalue()
        buffer.close()
        
        return pdf_bytes
    
    def _get_risk_style(self, risk_score: int) -> str:
        """Get the appropriate style based on risk score."""
        if risk_score >= 7:
            return 'RiskHigh'
        elif risk_score >= 4:
            return 'RiskMedium'
        else:
            return 'RiskLow'
    
    def _generate_recommendations(self, evaluation_data: Dict[str, Any]) -> list:
        """Generate recommendations based on the evaluation."""
        recommendations = []
        
        if not evaluation_data.get('security_assessment'):
            return ["Complete a full security assessment for this service"]
        
        assessment = evaluation_data['security_assessment']
        risk_score = assessment.get('risk_score', 0)
        
        if risk_score >= 7:
            recommendations.append("⚠️ HIGH RISK: Immediate security review recommended")
            recommendations.append("Implement additional access controls and monitoring")
        
        if assessment.get('risk_profile'):
            profile = assessment['risk_profile']
            
            if profile.get('data_exposure', 0) >= 7:
                recommendations.append("Implement data encryption at rest and in transit")
                recommendations.append("Review and restrict data access permissions")
            
            if profile.get('misconfiguration', 0) >= 6:
                recommendations.append("Conduct configuration audit against security best practices")
                recommendations.append("Enable configuration monitoring and drift detection")
            
            if profile.get('attack_surface', 0) >= 6:
                recommendations.append("Minimize exposed endpoints and services")
                recommendations.append("Implement network segmentation and firewall rules")
            
            if profile.get('compliance_violation', 0) >= 5:
                recommendations.append("Review compliance requirements for your industry")
                recommendations.append("Implement compliance monitoring and reporting")
        
        if 'public' in assessment.get('network_exposure', '').lower():
            recommendations.append("Consider using private endpoints or VPC Service Controls")
        
        if not assessment.get('compliance_certifications'):
            recommendations.append("Verify compliance certifications for regulatory requirements")
        
        if not recommendations:
            recommendations.append("Service appears well-configured. Continue monitoring for changes.")
        
        return recommendations
    
    def _generate_simple_pdf(self, evaluation_data: Dict[str, Any]) -> bytes:
        """
        Generate a comprehensive HTML-based report styled as PDF.
        """
        assessment = evaluation_data.get('security_assessment', {})
        risk_score = assessment.get('risk_score', 0)
        risk_level = self._get_risk_level(risk_score)
        risk_profile = assessment.get('risk_profile', {})
        
        # Build use cases HTML
        use_cases_html = ""
        if evaluation_data.get('use_cases'):
            use_cases_items = "".join([f"<li>{uc}</li>" for uc in evaluation_data['use_cases']])
            use_cases_html = f"""
            <div class="section">
                <h2>Use Cases</h2>
                <ul>{use_cases_items}</ul>
            </div>
            """
        
        # Build compliance HTML
        compliance_html = ""
        if assessment.get('compliance_certifications'):
            cert_items = "".join([f"<li>✓ {cert}</li>" for cert in assessment['compliance_certifications']])
            compliance_html = f"""
            <div class="section">
                <h3>Compliance Certifications</h3>
                <ul class="compliance-list">{cert_items}</ul>
            </div>
            """
        
        # Build IAM permissions HTML
        iam_html = ""
        if assessment.get('iam_permissions'):
            perms = assessment['iam_permissions'][:10]
            perm_items = "".join([f"<li><code>{p}</code></li>" for p in perms])
            more_text = f"<p><em>... and {len(assessment['iam_permissions']) - 10} more permissions</em></p>" if len(assessment['iam_permissions']) > 10 else ""
            iam_html = f"""
            <div class="section">
                <h3>Required IAM Permissions</h3>
                <ul class="permission-list">{perm_items}</ul>
                {more_text}
            </div>
            """
        
        # Generate recommendations
        recommendations = self._generate_recommendations(evaluation_data)
        rec_items = "".join([f"<li>{rec}</li>" for rec in recommendations])
        
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="UTF-8">
            <title>GCP Security Evaluation Report - {evaluation_data.get('service_name', 'Unknown')}</title>
            <style>
                @page {{ size: A4; margin: 2cm; }}
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    line-height: 1.6;
                    color: #333;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                    background: white;
                }}
                h1 {{
                    color: #1a73e8;
                    border-bottom: 3px solid #1a73e8;
                    padding-bottom: 10px;
                    margin-bottom: 30px;
                }}
                h2 {{
                    color: #34495e;
                    margin-top: 30px;
                    border-bottom: 1px solid #ecf0f1;
                    padding-bottom: 5px;
                }}
                h3 {{
                    color: #2c3e50;
                    margin-top: 20px;
                }}
                .header-info {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 5px;
                    margin-bottom: 30px;
                }}
                .risk-high {{
                    color: #e74c3c;
                    font-weight: bold;
                    font-size: 1.2em;
                }}
                .risk-medium {{
                    color: #f39c12;
                    font-weight: bold;
                    font-size: 1.2em;
                }}
                .risk-low {{
                    color: #27ae60;
                    font-weight: bold;
                    font-size: 1.2em;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                    margin: 20px 0;
                    box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 12px;
                    text-align: left;
                }}
                th {{
                    background-color: #3498db;
                    color: white;
                    font-weight: bold;
                }}
                tr:nth-child(even) {{
                    background-color: #f8f9fa;
                }}
                .info-table th {{
                    background-color: #95a5a6;
                    width: 30%;
                }}
                .risk-table th {{
                    background-color: #e74c3c;
                }}
                .section {{
                    margin: 30px 0;
                    padding: 20px;
                    background: #f8f9fa;
                    border-left: 4px solid #3498db;
                    border-radius: 0 5px 5px 0;
                }}
                ul {{
                    margin: 10px 0;
                    padding-left: 25px;
                }}
                li {{
                    margin: 5px 0;
                }}
                code {{
                    background: #ecf0f1;
                    padding: 2px 6px;
                    border-radius: 3px;
                    font-family: 'Courier New', monospace;
                    font-size: 0.9em;
                }}
                .compliance-list {{
                    list-style-type: none;
                    padding-left: 0;
                }}
                .compliance-list li {{
                    padding: 5px 0;
                    color: #27ae60;
                }}
                .threat-model {{
                    background: #fff3cd;
                    border: 1px solid #ffc107;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .recommendations {{
                    background: #d4edda;
                    border: 1px solid #c3e6cb;
                    padding: 15px;
                    border-radius: 5px;
                    margin: 20px 0;
                }}
                .footer {{
                    margin-top: 50px;
                    padding-top: 20px;
                    border-top: 1px solid #ecf0f1;
                    text-align: center;
                    color: #7f8c8d;
                    font-size: 0.9em;
                }}
                @media print {{
                    body {{ margin: 0; }}
                    .section {{ page-break-inside: avoid; }}
                }}
            </style>
        </head>
        <body>
            <h1>🔐 GCP Security Evaluation Report</h1>
            
            <div class="header-info">
                <strong>Report Generated:</strong> {datetime.now().strftime('%B %d, %Y at %I:%M %p')}<br>
                <strong>Service Evaluated:</strong> {evaluation_data.get('service_name', 'Unknown')}<br>
                <strong>Project ID:</strong> {evaluation_data.get('project_id', 'test-project')}
            </div>
            
            <h2>Executive Summary</h2>
            <p>This security evaluation report provides a comprehensive assessment of the <strong>{evaluation_data.get('service_name', 'Unknown')}</strong> service in Google Cloud Platform.</p>
            <p class="risk-{risk_level}">Overall Risk Score: {risk_score}/10</p>
            
            <h2>Service Information</h2>
            <table class="info-table">
                <tr>
                    <th>Property</th>
                    <th>Value</th>
                </tr>
                <tr>
                    <td><strong>Service Name</strong></td>
                    <td>{evaluation_data.get('service_name', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>Description</strong></td>
                    <td>{evaluation_data.get('description', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>Release Stage</strong></td>
                    <td>{evaluation_data.get('release_stage', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>Service Status</strong></td>
                    <td>{'✅ Enabled' if evaluation_data.get('is_enabled') else '❌ Not Enabled'}</td>
                </tr>
            </table>
            
            {use_cases_html}
            
            <h2>Security Assessment</h2>
            
            <h3>Risk Profile Analysis</h3>
            <table class="risk-table">
                <tr>
                    <th>Risk Category</th>
                    <th>Score (0-10)</th>
                    <th>Risk Level</th>
                </tr>
                <tr>
                    <td>Data Exposure</td>
                    <td>{risk_profile.get('data_exposure', 0)}</td>
                    <td>{self._get_risk_badge(risk_profile.get('data_exposure', 0))}</td>
                </tr>
                <tr>
                    <td>Misconfiguration</td>
                    <td>{risk_profile.get('misconfiguration', 0)}</td>
                    <td>{self._get_risk_badge(risk_profile.get('misconfiguration', 0))}</td>
                </tr>
                <tr>
                    <td>Attack Surface</td>
                    <td>{risk_profile.get('attack_surface', 0)}</td>
                    <td>{self._get_risk_badge(risk_profile.get('attack_surface', 0))}</td>
                </tr>
                <tr>
                    <td>Compliance Violation</td>
                    <td>{risk_profile.get('compliance_violation', 0)}</td>
                    <td>{self._get_risk_badge(risk_profile.get('compliance_violation', 0))}</td>
                </tr>
            </table>
            
            <h3>Security Configuration</h3>
            <table class="info-table">
                <tr>
                    <td><strong>Network Exposure</strong></td>
                    <td>{assessment.get('network_exposure', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>Data Encryption</strong></td>
                    <td>{assessment.get('data_encryption', 'N/A')}</td>
                </tr>
                <tr>
                    <td><strong>Data Residency</strong></td>
                    <td>{assessment.get('data_residency', 'N/A')}</td>
                </tr>
            </table>
            
            {compliance_html}
            
            {iam_html}
            
            <div class="threat-model">
                <h3>⚠️ Threat Model Summary</h3>
                <p>{assessment.get('threat_model_summary', 'No threat model available.')}</p>
            </div>
            
            <div class="recommendations">
                <h2>📋 Recommendations</h2>
                <ol>
                    {rec_items}
                </ol>
            </div>
            
            <div class="footer">
                <p>This report was generated automatically by the GCP Security Agent</p>
                <p>For questions or concerns, contact your security team</p>
                <p>© 2024 Security Evaluation System - Confidential</p>
            </div>
        </body>
        </html>
        """
        
        return html_content.encode('utf-8')
    
    def _get_risk_badge(self, score: int) -> str:
        """Get risk badge HTML based on score."""
        if score >= 7:
            return '<span style="color: #e74c3c; font-weight: bold;">HIGH</span>'
        elif score >= 4:
            return '<span style="color: #f39c12; font-weight: bold;">MEDIUM</span>'
        else:
            return '<span style="color: #27ae60; font-weight: bold;">LOW</span>'
    
    def _get_risk_level(self, score: int) -> str:
        """Get risk level string."""
        if score >= 7:
            return 'high'
        elif score >= 4:
            return 'medium'
        else:
            return 'low'


# Singleton instance
pdf_generator = PDFReportGenerator()