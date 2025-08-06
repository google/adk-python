"""Pydantic models for compliance feature.

This module defines the data models used for compliance evaluation requests
and responses. It includes models for various compliance frameworks such as
SOC2, ISO27001, GDPR, HIPAA, and PCI_DSS.

Classes:
    ComplianceEvaluationRequest: Request model for initiating compliance evaluations
    ComplianceEvaluationResponse: Response model containing evaluation results
    
Examples:
    Creating a compliance evaluation request:
        request = ComplianceEvaluationRequest(
            project_id="my-project-123",
            framework="SOC2"
        )
        
    Processing evaluation response:
        if response.compliant:
            print("Project is compliant!")
        else:
            print(f"Found {len(response.findings)} compliance issues")
"""

from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional


class ComplianceEvaluationRequest(BaseModel):
    """Request model for compliance evaluation.
    
    This model represents a request to evaluate a GCP project's compliance
    against one or more security frameworks.
    
    Attributes:
        project_id (str): The GCP project ID to evaluate
        api_name (Optional[str]): Specific API to evaluate (if None, evaluates entire project)
        framework (str): Primary compliance framework (SOC2, ISO27001, GDPR, HIPAA, PCI_DSS) 
        frameworks (Optional[List[str]]): Additional frameworks to evaluate against
        
    Example:
        request = ComplianceEvaluationRequest(
            project_id="my-gcp-project",
            framework="SOC2",
            frameworks=["ISO27001", "GDPR"]
        )
    """
    project_id: str
    api_name: Optional[str] = None
    framework: str = "SOC2"
    frameworks: Optional[List[str]] = None


class ComplianceEvaluationResponse(BaseModel):
    """Response model for compliance evaluation results.
    
    This model contains the results of a compliance evaluation, including
    the overall compliance status, detailed findings, and recommendations
    for remediation.
    
    Attributes:
        resource_type (str): Type of resource that was evaluated
        framework (str): The compliance framework that was used
        compliant (bool): Whether the resource meets compliance requirements
        findings (List[Dict[str, Any]]): Detailed findings and violations
        recommendations (List[str]): Recommendations for achieving compliance
        
    Example:
        if response.compliant:
            print("✅ Compliance requirements met")
        else:
            print(f"❌ Found {len(response.findings)} compliance issues")
            for rec in response.recommendations:
                print(f"- {rec}")
    """
    resource_type: str
    framework: str
    compliant: bool
    findings: List[Dict[str, Any]]
    recommendations: List[str]