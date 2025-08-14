"""
Smart Query Router for ADK Security Agent
Routes queries to appropriate specialists based on intent detection
"""

import re
from typing import Dict, List, Tuple, Optional
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Specialist(Enum):
    """Available specialist agents"""
    STORAGE = "StorageSecuritySpecialist"
    IAM = "IAMSecuritySpecialist"
    NETWORK = "NetworkSecuritySpecialist"
    COMPLIANCE = "ComplianceSpecialist"
    FINOPS = "FinOpsSpecialist"
    COMPUTE = "ComputeSecuritySpecialist"
    DATABASE = "DatabaseSecuritySpecialist"
    KUBERNETES = "KubernetesSpecialist"
    MONITORING = "MonitoringSpecialist"
    GENERAL = "GeneralSecurityAgent"

class QueryRouter:
    """Routes queries to appropriate specialists based on keyword and pattern matching"""
    
    def __init__(self):
        # Define keyword mappings for each specialist
        self.keyword_mappings = {
            Specialist.STORAGE: {
                "keywords": ["bucket", "storage", "gcs", "blob", "object", "archive", 
                            "backup", "lifecycle", "retention", "versioning"],
                "patterns": [
                    r".*storage.*security.*",
                    r".*bucket.*public.*",
                    r".*gcs.*access.*",
                    r".*backup.*policy.*"
                ],
                "weight": 1.0
            },
            Specialist.IAM: {
                "keywords": ["iam", "permission", "role", "identity", "access", "user",
                            "service account", "privilege", "authorization", "authentication",
                            "mfa", "2fa", "group", "member"],
                "patterns": [
                    r".*who has access.*",
                    r".*permission.*review.*",
                    r".*service account.*",
                    r".*role.*assignment.*",
                    r".*user.*access.*",
                    r".*privilege.*escalation.*"
                ],
                "weight": 1.0
            },
            Specialist.NETWORK: {
                "keywords": ["firewall", "network", "vpc", "subnet", "route", "ip",
                            "ingress", "egress", "load balancer", "cdn", "dns",
                            "nat", "gateway", "peering", "interconnect", "vpn"],
                "patterns": [
                    r".*firewall.*rule.*",
                    r".*network.*security.*",
                    r".*vpc.*configuration.*",
                    r".*open port.*",
                    r".*traffic.*analysis.*",
                    r".*network.*segmentation.*"
                ],
                "weight": 1.0
            },
            Specialist.COMPLIANCE: {
                "keywords": ["compliance", "gdpr", "hipaa", "pci", "sox", "iso",
                            "cis", "benchmark", "audit", "regulation", "standard",
                            "framework", "control", "requirement", "certification"],
                "patterns": [
                    r".*compliance.*status.*",
                    r".*regulatory.*requirement.*",
                    r".*audit.*report.*",
                    r".*compliance.*framework.*",
                    r".*certification.*status.*",
                    r".*cis.*benchmark.*"
                ],
                "weight": 1.0
            },
            Specialist.FINOPS: {
                "keywords": ["cost", "billing", "budget", "spend", "price", "expense",
                            "optimization", "savings", "waste", "rightsizing", "commitment",
                            "discount", "credit", "invoice", "forecast"],
                "patterns": [
                    r".*cost.*optimization.*",
                    r".*reduce.*spend.*",
                    r".*billing.*analysis.*",
                    r".*budget.*alert.*",
                    r".*cost.*breakdown.*",
                    r".*unused.*resource.*",
                    r".*how much.*cost.*"
                ],
                "weight": 1.0
            },
            Specialist.COMPUTE: {
                "keywords": ["vm", "instance", "compute", "gce", "machine", "cpu",
                            "memory", "disk", "snapshot", "image", "template",
                            "ssh", "rdp", "startup", "metadata"],
                "patterns": [
                    r".*instance.*security.*",
                    r".*vm.*configuration.*",
                    r".*compute.*engine.*",
                    r".*ssh.*key.*",
                    r".*instance.*patch.*",
                    r".*metadata.*security.*"
                ],
                "weight": 0.9
            },
            Specialist.DATABASE: {
                "keywords": ["database", "sql", "mysql", "postgres", "mongodb", "redis",
                            "backup", "replication", "encryption", "ssl", "tls",
                            "query", "schema", "table", "index"],
                "patterns": [
                    r".*database.*security.*",
                    r".*sql.*injection.*",
                    r".*database.*backup.*",
                    r".*encryption.*rest.*",
                    r".*database.*access.*"
                ],
                "weight": 0.9
            },
            Specialist.KUBERNETES: {
                "keywords": ["kubernetes", "k8s", "gke", "pod", "deployment", "service",
                            "ingress", "namespace", "cluster", "node", "container",
                            "helm", "kubectl", "rbac", "workload"],
                "patterns": [
                    r".*kubernetes.*security.*",
                    r".*gke.*cluster.*",
                    r".*pod.*security.*",
                    r".*rbac.*configuration.*",
                    r".*workload.*identity.*"
                ],
                "weight": 0.9
            },
            Specialist.MONITORING: {
                "keywords": ["monitor", "alert", "log", "metric", "trace", "observability",
                            "dashboard", "notification", "threshold", "sla", "slo",
                            "uptime", "latency", "error rate"],
                "patterns": [
                    r".*monitoring.*setup.*",
                    r".*alert.*configuration.*",
                    r".*log.*analysis.*",
                    r".*metric.*collection.*",
                    r".*dashboard.*create.*"
                ],
                "weight": 0.8
            }
        }
        
        # Define question type patterns
        self.question_patterns = {
            "analysis": ["analyze", "review", "audit", "assess", "evaluate", "check", "inspect"],
            "listing": ["list", "show", "display", "get", "find", "what are", "which"],
            "fixing": ["fix", "remediate", "resolve", "patch", "secure", "harden", "improve"],
            "explanation": ["explain", "what is", "how does", "why", "describe", "tell me about"],
            "configuration": ["configure", "setup", "enable", "disable", "implement", "deploy"]
        }
    
    def detect_specialist(self, query: str) -> Tuple[Specialist, float, Dict[str, any]]:
        """
        Detect which specialist should handle the query
        Returns: (Specialist, confidence_score, context)
        """
        query_lower = query.lower()
        scores = {}
        matched_keywords = {}
        
        # Score each specialist based on keyword and pattern matching
        for specialist, config in self.keyword_mappings.items():
            score = 0.0
            keywords_found = []
            
            # Check keywords
            for keyword in config["keywords"]:
                if keyword in query_lower:
                    score += 1.0 * config["weight"]
                    keywords_found.append(keyword)
            
            # Check patterns
            for pattern in config["patterns"]:
                if re.search(pattern, query_lower):
                    score += 2.0 * config["weight"]  # Patterns get higher weight
            
            scores[specialist] = score
            matched_keywords[specialist] = keywords_found
        
        # Get the best match
        best_specialist = max(scores.items(), key=lambda x: x[1])
        
        # If no clear match, use general agent
        if best_specialist[1] < 0.5:
            return Specialist.GENERAL, 0.5, {"matched_keywords": [], "query_type": self.detect_query_type(query)}
        
        # Calculate confidence (normalize to 0-1)
        confidence = min(best_specialist[1] / 5.0, 1.0)
        
        context = {
            "matched_keywords": matched_keywords[best_specialist[0]],
            "query_type": self.detect_query_type(query),
            "all_scores": scores
        }
        
        return best_specialist[0], confidence, context
    
    def detect_query_type(self, query: str) -> str:
        """Detect the type of query (analysis, listing, fixing, etc.)"""
        query_lower = query.lower()
        
        for query_type, patterns in self.question_patterns.items():
            for pattern in patterns:
                if pattern in query_lower:
                    return query_type
        
        return "general"
    
    def get_routing_explanation(self, specialist: Specialist, confidence: float, context: Dict) -> str:
        """Generate an explanation for why this specialist was chosen"""
        keywords = context.get("matched_keywords", [])
        query_type = context.get("query_type", "general")
        
        if specialist == Specialist.GENERAL:
            return "Using general security agent for broad analysis"
        
        explanation = f"Routing to {specialist.value} (confidence: {confidence:.0%}) "
        explanation += f"for {query_type} request"
        
        if keywords:
            explanation += f" based on keywords: {', '.join(keywords[:3])}"
        
        return explanation
    
    def route_query(self, query: str) -> Dict[str, any]:
        """
        Main routing function that returns routing decision
        """
        specialist, confidence, context = self.detect_specialist(query)
        explanation = self.get_routing_explanation(specialist, confidence, context)
        
        logger.info(f"Query routing: {explanation}")
        
        return {
            "specialist": specialist,
            "confidence": confidence,
            "context": context,
            "explanation": explanation,
            "endpoint": self.get_specialist_endpoint(specialist)
        }
    
    def get_specialist_endpoint(self, specialist: Specialist) -> str:
        """Get the API endpoint for the specialist"""
        endpoint_map = {
            Specialist.STORAGE: "/api/v1/storage/analyze",
            Specialist.IAM: "/api/v1/iam/analyze",
            Specialist.NETWORK: "/api/v1/network/analyze",
            Specialist.COMPLIANCE: "/api/v1/compliance/analyze",
            Specialist.FINOPS: "/api/v1/cost/analyze",
            Specialist.COMPUTE: "/api/v1/compute/analyze",
            Specialist.DATABASE: "/api/v1/database/analyze",
            Specialist.KUBERNETES: "/api/v1/k8s/analyze",
            Specialist.MONITORING: "/api/v1/monitoring/analyze",
            Specialist.GENERAL: "/api/v1/security/analyze"
        }
        return endpoint_map.get(specialist, "/api/v1/security/analyze")

# Global router instance
query_router = QueryRouter()