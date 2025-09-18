"""
Prompt Templates for Frontend Agents
====================================

Contains all prompt templates used by frontend agents for:
- Query analysis and classification
- Query enhancement and optimization
- Tool selection and routing
"""

from typing import Dict, Any

class PromptTemplates:
    """
    Collection of prompt templates for frontend agents.
    """
    
    # Base instruction for query analysis
    QUERY_ANALYZER_INSTRUCTION = """
You are an intelligent query analyzer for a GCP security agent system.

Your role is to analyze user queries and determine:
1. The type of query (data lookup, help request, clarification, search)
2. Whether it requires backend database access
3. How to enhance the query for optimal results
4. Which specific tool/query_type would be most effective

For data queries, suggest specific query types based on these categories:
- storage_buckets: For Cloud Storage security, encryption, bucket policies
- security_findings: For Security Command Center findings, vulnerabilities
- iam_accounts: For IAM users, roles, permissions, service accounts
- networks: For VPC, firewall rules, network security
- assets: For general asset inventory and discovery
- org_policies: For organization policies and constraints
- compliance: For compliance posture and recommendations

Enhance vague queries by:
- Adding specific context from conversation history
- Clarifying ambiguous terms
- Suggesting specific data to retrieve
- Adding relevant security focus areas

Always return valid JSON with this structure:
{
    "query_type": "data|help|clarification|search",
    "needs_backend": true|false,
    "enhanced_query": "improved version with context",
    "suggested_tool": "specific query_type if applicable",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation of analysis"
}
"""
    
    @staticmethod
    def build_analysis_prompt(current_query: str, context: str) -> str:
        """
        Build a complete prompt for query analysis.
        
        Args:
            current_query: The user's current query
            context: Conversation context from recent messages
            
        Returns:
            Complete prompt for the LLM
        """
        return f"""
{PromptTemplates.QUERY_ANALYZER_INSTRUCTION}

Conversation Context:
{context}

Current User Query: "{current_query}"

Analyze this query considering the conversation context. Focus on:
1. What specific GCP security information is the user seeking?
2. Can this be answered with cached knowledge or needs fresh data?
3. How can we make the query more specific and actionable?
4. What tool would best serve this request?

Respond with JSON only - no additional text.
"""
    
    # Tool selection guidance
    TOOL_SELECTION_GUIDE = {
        'storage_buckets': {
            'keywords': ['bucket', 'storage', 'encrypt', 'gcs', 'cloud storage'],
            'description': 'Use for Cloud Storage bucket security analysis',
            'example_queries': [
                'Are my storage buckets encrypted?',
                'Show me bucket policies',
                'Which buckets are publicly accessible?'
            ]
        },
        'security_findings': {
            'keywords': ['finding', 'vulnerability', 'security', 'threat', 'alert'],
            'description': 'Use for Security Command Center findings',
            'example_queries': [
                'What security findings do I have?',
                'Show critical vulnerabilities',
                'Recent security alerts'
            ]
        },
        'iam_accounts': {
            'keywords': ['iam', 'user', 'role', 'permission', 'service account'],
            'description': 'Use for IAM analysis and user management',
            'example_queries': [
                'List IAM users with admin access',
                'Show service accounts',
                'Who has access to project X?'
            ]
        },
        'networks': {
            'keywords': ['network', 'vpc', 'firewall', 'subnet', 'connectivity'],
            'description': 'Use for network security analysis',
            'example_queries': [
                'Show firewall rules',
                'Network security posture',
                'VPC configuration'
            ]
        },
        'assets': {
            'keywords': ['asset', 'inventory', 'resource', 'discovery'],
            'description': 'Use for general asset discovery and inventory',
            'example_queries': [
                'What assets do I have?',
                'Show all resources',
                'Asset inventory'
            ]
        }
    }
    
    @staticmethod
    def get_enhancement_prompt(query: str, analysis: Dict[str, Any]) -> str:
        """
        Build a prompt for query enhancement.
        
        Args:
            query: Original query
            analysis: Analysis results
            
        Returns:
            Enhanced query prompt
        """
        suggested_tool = analysis.get('suggested_tool')
        
        if suggested_tool and suggested_tool in PromptTemplates.TOOL_SELECTION_GUIDE:
            tool_info = PromptTemplates.TOOL_SELECTION_GUIDE[suggested_tool]
            enhancement = f"""
Based on your query about {query}, I'll search {tool_info['description'].lower()}.

To provide the most relevant results, I'll focus on:
- {tool_info['description']}
- Current security status and configurations
- Any potential security concerns or recommendations

Original query: {query}
"""
            return enhancement
        
        return query
    
    # Local knowledge prompts
    LOCAL_KNOWLEDGE_PROMPTS = {
        'help': """
I can help you with GCP security analysis! Here are some things you can ask me:

**Security Analysis:**
- "Show me security findings"
- "Are my storage buckets encrypted?"
- "What IAM users have admin access?"

**Best Practices:**
- "GCP security best practices"
- "How to improve my security posture?"
- "Encryption recommendations"

**Specific Queries:**
- "Firewall rules analysis"
- "Service account permissions"
- "Public bucket exposure"

What would you like to explore?
""",
        
        'capabilities': """
I'm your GCP Security Assistant with access to:

**Real-time Data Analysis:**
- Security Command Center findings
- IAM accounts and permissions
- Cloud Storage bucket configurations
- Network security settings
- Asset inventory

**Intelligent Features:**
- Contextual query understanding
- Security recommendations
- Best practice guidance
- Risk assessment

**How I Work:**
1. Analyze your query with conversation context
2. Check local knowledge for quick answers
3. Query your GCP environment for current data
4. Provide actionable insights and recommendations

Try asking me about your security posture!
"""
    }
    
    @staticmethod
    def get_local_response(query_type: str) -> str:
        """
        Get a local knowledge response for common queries.
        
        Args:
            query_type: Type of local query (help, capabilities, etc.)
            
        Returns:
            Local response text
        """
        return PromptTemplates.LOCAL_KNOWLEDGE_PROMPTS.get(
            query_type, 
            "I can help with GCP security analysis. What would you like to know?"
        )
    
    # Context building templates
    @staticmethod
    def build_context_string(conversation_history) -> str:
        """
        Build a context string from conversation history.
        
        Args:
            conversation_history: List of recent messages
            
        Returns:
            Formatted context string
        """
        if not conversation_history:
            return "No previous conversation context."
        
        context_lines = []
        for i, msg in enumerate(conversation_history[-3:], 1):  # Last 3 messages
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')[:150]  # Truncate for context
            context_lines.append(f"{i}. {role.title()}: {content}...")
        
        return "\n".join(context_lines)
    
    # Error handling prompts
    ERROR_RESPONSES = {
        'api_failure': "I'm having trouble analyzing your query right now. I'll send it directly to the backend for processing.",
        'parsing_error': "I couldn't parse the query analysis, but I'll still try to help with your request.",
        'enhancement_error': "Query enhancement failed, but I'll process your original question.",
        'backend_error': "I'm having trouble connecting to the backend. Please try again in a moment."
    }
    
    @staticmethod
    def get_error_response(error_type: str) -> str:
        """
        Get an appropriate error response.
        
        Args:
            error_type: Type of error that occurred
            
        Returns:
            User-friendly error message
        """
        return PromptTemplates.ERROR_RESPONSES.get(
            error_type,
            "I encountered an unexpected error. Please try again."
        )
