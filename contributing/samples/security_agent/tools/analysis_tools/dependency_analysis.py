"""
Dependency Analysis Tools

Tools for building dependency graphs and analyzing risk propagation through service dependencies.
"""

import json
from typing import Dict, Any


def get_api_dependency_graph(api_name: str, kb_path: str) -> dict:
    """Recursively build a dependency graph for the given API.
    
    This function traverses the dependency tree starting from the specified API
    and builds a nested dictionary representing the dependency relationships.
    It handles circular dependencies by tracking visited nodes.
    
    Args:
        api_name: Name of the API to build the dependency graph for.
        kb_path: Path to the knowledge base JSON file.
        
    Returns:
        Nested dictionary representing the dependency graph structure.
        Format: {api_name: {dependency: {sub_dependency: {}}}}
        
    Example:
        >>> graph = get_api_dependency_graph("Cloud Storage", kb_path)
        >>> print(graph)
        {'Cloud Storage': {'IAM': {}, 'Cloud KMS': {'IAM': {}}}}
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    api_map = {api['name']: api for api in kb['apis']}
    visited = set()
    def build_graph(name):
        if name not in api_map or name in visited:
            return {}
        visited.add(name)
        deps = api_map[name].get('dependencies', [])
        return {name: {dep: build_graph(dep) for dep in deps}}
    return build_graph(api_name)


def propagate_risk(api_name: str, kb_path: str) -> dict:
    """Propagate risk through the dependency graph and report at-risk services.
    
    This function analyzes the dependency tree starting from the specified API
    and identifies all services that are at risk due to direct vulnerabilities
    or dependencies on vulnerable services. It provides detailed reasoning and
    the path of risk propagation.
    
    Args:
        api_name: Name of the API to analyze for risk propagation.
        kb_path: Path to the knowledge base JSON file.
        
    Returns:
        Dictionary mapping service names to risk information:
        {
            'service_name': {
                'at_risk': bool,
                'reason': str,
                'path': List[str]
            }
        }
        
    Example:
        >>> risk_report = propagate_risk("Cloud Storage", kb_path)
        >>> print(risk_report)
        {
            'Cloud Storage': {
                'at_risk': True,
                'reason': 'Depends on a vulnerable service.',
                'path': ['Cloud Storage', 'Cloud KMS']
            },
            'Cloud KMS': {
                'at_risk': True,
                'reason': 'Cloud KMS is directly vulnerable.',
                'path': ['Cloud Storage', 'Cloud KMS']
            }
        }
    """
    with open(kb_path, 'r') as f:
        kb = json.load(f)
    api_map = {api['name']: api for api in kb['apis']}
    risk_report = {}
    def check_risk(name, path=None):
        if path is None:
            path = [name]
        api = api_map.get(name)
        if not api:
            return False
        if api.get('vulnerable', False):
            risk_report[name] = {
                'at_risk': True,
                'reason': f"{name} is directly vulnerable.",
                'path': list(path)
            }
            return True
        deps = api.get('dependencies', [])
        at_risk = False
        for dep in deps:
            if check_risk(dep, path + [dep]):
                at_risk = True
        if at_risk:
            risk_report[name] = {
                'at_risk': True,
                'reason': f"Depends on a vulnerable service.",
                'path': list(path)
            }
        else:
            risk_report[name] = {
                'at_risk': False,
                'reason': "No vulnerable dependencies detected.",
                'path': list(path)
            }
        return at_risk
    check_risk(api_name)
    return risk_report