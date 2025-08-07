"""
Security Knowledge Base Tools

Tools for working with security knowledge bases, API evaluations, and documentation scraping.
"""

import json
import os
from typing import Dict, Any
from google.adk.tools.tool_context import ToolContext
import requests
from bs4 import BeautifulSoup


def load_security_kb(kb_path: str) -> Dict[str, Any]:
    """Load the GCP API security knowledge base from a JSON file.
    
    Args:
        kb_path: Path to the JSON knowledge base file.
        
    Returns:
        Dictionary containing the parsed knowledge base data.
        
    Raises:
        FileNotFoundError: If the knowledge base file doesn't exist.
        JSONDecodeError: If the JSON file is malformed.
    """
    with open(kb_path, 'r') as f:
        return json.load(f)


def evaluate_api_security(api_name: str, tool_context: ToolContext) -> str:
    """Evaluate the security stance of a GCP API using the knowledge base.

    This function looks up the specified API in the knowledge base and returns
    a formatted summary of security considerations and recommended practices.

    Args:
        api_name: Name of the GCP API to evaluate (case-insensitive).
        tool_context: ToolContext for state and logging (unused in this implementation).

    Returns:
        A formatted string containing:
        - Security evaluation summary
        - List of security considerations
        - List of recommended practices
        - Documentation URL reference
        
    Example:
        >>> result = evaluate_api_security("Cloud Storage", tool_context)
        >>> print(result)
        Security Evaluation for Cloud Storage (see docs: https://cloud.google.com/storage/docs):
        Security Considerations:
        - Data is encrypted at rest and in transit.
        - IAM roles control access to buckets and objects.
        ...
    """
    kb_path = os.path.join(os.path.dirname(__file__), '../../agents/gcp_api_security_kb.json')
    kb = load_security_kb(kb_path)
    api_info = next((api for api in kb['apis'] if api['name'].lower() == api_name.lower()), None)
    if not api_info:
        return f"No security information found for API: {api_name}. Please check the API name or update the knowledge base."
    summary = [
        f"Security Evaluation for {api_info['name']} (see docs: {api_info['documentation_url']}):",
        "\\nSecurity Considerations:",
    ]
    summary.extend(f"- {item}" for item in api_info['security_considerations'])
    summary.append("\\nRecommended Practices:")
    summary.extend(f"- {item}" for item in api_info['recommended_practices'])
    return '\\n'.join(summary)


def scrape_api_documentation(doc_url: str, tool_context: ToolContext = None) -> str:
    """Scrape the documentation URL for limits or considerations.
    
    This function fetches a web page and extracts text content that mentions
    security-related keywords like 'limit', 'limitation', 'consideration',
    'quota', or 'restriction'. It's useful for automatically gathering
    security information from official documentation.
    
    Args:
        doc_url: URL of the documentation page to scrape.
        tool_context: ToolContext for state and logging (unused in this implementation).
        
    Returns:
        String containing up to 20 findings that match the security keywords,
        or an error message if scraping fails.
        
    Example:
        >>> findings = scrape_api_documentation("https://cloud.google.com/storage/quotas")
        >>> print(findings)
        Findings from https://cloud.google.com/storage/quotas:
        - Storage quotas and limits
        - Request rate limits
        - Object size limitations
        ...
    """
    try:
        resp = requests.get(doc_url, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, 'html.parser')
        text = soup.get_text(separator='\\n')
        lines = text.splitlines()
        findings = []
        keywords = ['limit', 'limitation', 'consideration', 'quota', 'restriction']
        for line in lines:
            l = line.strip()
            if not l:
                continue
            if any(kw in l.lower() for kw in keywords):
                findings.append(l)
        if not findings:
            return f"No explicit limits or considerations found at {doc_url}."
        return f"Findings from {doc_url}:\\n" + '\\n'.join(findings[:20])
    except Exception as e:
        return f"Error scraping {doc_url}: {e}"