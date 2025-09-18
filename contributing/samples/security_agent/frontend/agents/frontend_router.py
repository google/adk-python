"""
Frontend Router Agent - Intelligent query preprocessing and routing.

This agent sits in the frontend and:
1. Analyzes user queries with conversation context
2. Enhances queries for better backend agent performance
3. Can handle certain queries locally
4. Routes complex queries to backend with enriched context
"""

import os
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# Conditional import for google.generativeai
try:
    import google.generativeai as genai
    GENAI_AVAILABLE = True
except ImportError:
    genai = None
    GENAI_AVAILABLE = False
    logger.warning("google.generativeai not available - frontend router will use fallback mode only")

@dataclass
class QueryAnalysis:
    """Analysis of a user query."""
    query_type: str  # 'data', 'help', 'clarification', 'search'
    needs_backend: bool
    enhanced_query: str
    suggested_tool: Optional[str] = None
    confidence: float = 0.0

class FrontendRouterAgent:
    """Frontend agent that preprocesses and routes queries."""

    def __init__(self):
        """Initialize the frontend router agent."""
        from frontend.utils.config import FrontendConfig
        from frontend.agents.prompts import PromptTemplates

        # Load configuration
        self.config = FrontendConfig.get_frontend_agent_config()

        # Configure Gemini for frontend analysis
        api_key = self.config.get('gemini_api_key')
        if api_key and self.config.get('router_enabled', True) and GENAI_AVAILABLE:
            genai.configure(api_key=api_key)
            model_name = self.config.get('router_model', 'gemini-1.5-flash')
            self.model = genai.GenerativeModel(model_name)
            self.enabled = True
        else:
            self.model = None
            self.enabled = False
            if not GENAI_AVAILABLE:
                logger.warning("Frontend router disabled - google.generativeai not available")
            elif not api_key:
                logger.warning("Frontend router disabled - no API key found")
            else:
                logger.warning("Frontend router disabled - disabled in config")

        # Use prompt templates
        self.instruction = PromptTemplates.QUERY_ANALYZER_INSTRUCTION
        self.prompt_templates = PromptTemplates()

    def analyze_query(self,
                     current_query: str,
                     conversation_history: List[Dict[str, str]] = None) -> QueryAnalysis:
        """
        Analyze a query with conversation context.

        Args:
            current_query: The user's current query
            conversation_history: Last 2-3 messages for context

        Returns:
            QueryAnalysis with routing decision
        """
        if not self.enabled or not self.model:
            # Fallback to simple analysis
            return self._simple_analysis(current_query)

        try:
            # Build context from conversation history
            context = self.prompt_templates.build_context_string(conversation_history)

            # Use template to build the full prompt
            prompt = self.prompt_templates.build_analysis_prompt(current_query, context)

            response = self.model.generate_content(prompt)

            # Parse response
            import json
            try:
                result = json.loads(response.text)
                # Log enhancement if debugging is enabled
                if self.config.get('log_enhancements', False):
                    logger.info(f"Query analysis: {result}")

                return QueryAnalysis(**result)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse analysis response: {e}")
                # If parsing fails, use simple analysis
                return self._simple_analysis(current_query)

        except Exception as e:
            logger.error(f"Frontend agent error: {e}")
            return self._simple_analysis(current_query)

    def _build_context(self, history: List[Dict[str, str]]) -> str:
        """Build context string from conversation history."""
        if not history:
            return "No previous context"

        # Take last 2 messages for context
        recent = history[-2:] if len(history) >= 2 else history

        context_lines = []
        for msg in recent:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")[:200]  # Truncate long messages
            context_lines.append(f"{role}: {content}")

        return "\n".join(context_lines)

    def _simple_analysis(self, query: str) -> QueryAnalysis:
        """Simple keyword-based analysis fallback."""
        query_lower = query.lower()

        # Check for data queries
        data_keywords = {
            'bucket': 'storage_buckets',
            'storage': 'storage_buckets',
            'finding': 'security_findings',
            'security': 'security_findings',
            'iam': 'iam_accounts',
            'user': 'iam_accounts',
            'encrypt': 'storage_buckets',
            'network': 'networks',
            'asset': 'assets'
        }

        for keyword, tool in data_keywords.items():
            if keyword in query_lower:
                return QueryAnalysis(
                    query_type='data',
                    needs_backend=True,
                    enhanced_query=query,
                    suggested_tool=tool,
                    confidence=0.7
                )

        # Check for help queries
        help_keywords = ['how', 'what', 'why', 'explain', 'help']
        if any(kw in query_lower for kw in help_keywords):
            return QueryAnalysis(
                query_type='help',
                needs_backend=True,
                enhanced_query=query,
                confidence=0.6
            )

        # Default to backend routing
        return QueryAnalysis(
            query_type='search',
            needs_backend=True,
            enhanced_query=query,
            confidence=0.5
        )

    def enhance_for_backend(self,
                           query: str,
                           analysis: QueryAnalysis) -> str:
        """
        Enhance query specifically for backend agent.

        Args:
            query: Original query
            analysis: Query analysis result

        Returns:
            Enhanced query optimized for backend agent
        """
        # Use prompt template for enhancement
        enhanced_query = self.prompt_templates.get_enhancement_prompt(query, analysis.__dict__)

        if analysis.suggested_tool:
            # Add explicit instruction for tool usage
            tool_instruction = f"IMPORTANT: Use query_security_data with query_type='{analysis.suggested_tool}'. "
            enhanced_query = tool_instruction + enhanced_query

        # Log enhancement if debugging is enabled
        if self.config.get('log_enhancements', False) and enhanced_query != query:
            logger.info(f"Enhanced query from: {query}")
            logger.info(f"To: {enhanced_query}")

        return enhanced_query


class LocalLookupAgent:
    """
    Local agent that can handle certain queries without backend.
    Useful for help text, documentation, and cached responses.
    """

    def __init__(self):
        """Initialize local lookup agent."""
        from frontend.agents.prompts import PromptTemplates

        self.prompt_templates = PromptTemplates()
        self.local_knowledge = {
            'encryption': {
                'response': """To encrypt data in GCP:
                1. **Storage Buckets**: Enable default encryption with Google-managed or customer-managed keys
                2. **At Rest**: All GCP services encrypt data at rest by default
                3. **In Transit**: Use HTTPS/TLS for all communications
                4. **CMEK**: Use Cloud KMS for customer-managed encryption keys

                For specific bucket encryption status, I'll need to query the backend.""",
                'needs_followup': True
            },
            'best practices': {
                'response': """GCP Security Best Practices:
                1. Enable MFA for all users
                2. Use least privilege IAM policies
                3. Enable audit logging
                4. Regular security assessments
                5. Use VPC Service Controls
                6. Enable Security Command Center

                For your specific environment analysis, let me check the backend.""",
                'needs_followup': True
            },
            'help': {
                'response': self.prompt_templates.get_local_response('help'),
                'needs_followup': False
            },
            'capabilities': {
                'response': self.prompt_templates.get_local_response('capabilities'),
                'needs_followup': False
            }
        }

    def can_handle_locally(self, query: str) -> bool:
        """Check if query can be handled locally."""
        query_lower = query.lower()

        # Check for direct topic matches
        for topic in self.local_knowledge.keys():
            if topic in query_lower:
                return True

        # Check for additional patterns
        help_patterns = ['what can you do', 'what are you', 'capabilities', 'what do you do']
        if any(pattern in query_lower for pattern in help_patterns):
            return True

        return False

    def handle_query(self, query: str) -> Dict[str, Any]:
        """Handle a query locally if possible."""
        query_lower = query.lower()

        # Check direct topic matches first
        for topic, info in self.local_knowledge.items():
            if topic in query_lower:
                return {
                    'success': True,
                    'response': info['response'],
                    'needs_backend': info.get('needs_followup', False),
                    'source': 'local_cache'
                }

        # Check for help/capabilities patterns
        help_patterns = ['what can you do', 'what are you', 'capabilities', 'what do you do']
        if any(pattern in query_lower for pattern in help_patterns):
            return {
                'success': True,
                'response': self.local_knowledge['capabilities']['response'],
                'needs_backend': False,
                'source': 'local_cache'
            }

        return {
            'success': False,
            'needs_backend': True
        }