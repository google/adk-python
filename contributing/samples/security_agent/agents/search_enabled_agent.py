"""
Search-Enabled Security Agent using Gemini's Google Search Grounding
Implements the proper way to use Google Search with Gemini API
"""

import os
import logging
from typing import Dict, List, Optional, Any
import vertexai
from vertexai.generative_models import (
    GenerativeModel, 
    Tool,
    grounding,
    GenerationConfig,
    SafetySetting,
    HarmCategory,
    HarmBlockThreshold
)

logger = logging.getLogger(__name__)


class SearchEnabledSecurityAgent:
    """Security agent with Google Search grounding capabilities using Gemini API"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        """
        Initialize the search-enabled agent with Gemini and Google Search grounding.
        
        Args:
            project_id: GCP project ID
            location: Vertex AI location
        """
        self.project_id = project_id
        self.location = location
        
        # Initialize Vertex AI
        vertexai.init(project=project_id, location=location)
        
        # Create model with Google Search grounding
        self.model = self._create_search_enabled_model()
        
        logger.info(f"✅ Search-enabled agent initialized for project: {project_id}")
    
    def _create_search_enabled_model(self) -> GenerativeModel:
        """
        Create a Gemini model with Google Search grounding enabled.
        
        Returns:
            GenerativeModel configured with Google Search
        """
        # Configure Google Search grounding
        google_search_tool = Tool.from_google_search_retrieval(
            grounding.GoogleSearchRetrieval()
        )
        
        # Create model with search grounding
        model = GenerativeModel(
            model_name="gemini-1.5-pro",  # or "gemini-1.5-flash" for faster responses
            tools=[google_search_tool],
            system_instruction="""You are a helpful security expert assistant with access to Google Search.
            
When answering questions:
1. Use Google Search to find the most current and accurate information
2. Focus on security best practices and GCP-specific guidance
3. Cite sources when providing information from search results
4. Distinguish between general knowledge and searched information
5. Provide actionable recommendations based on the latest information

For security-related queries:
- Always search for the latest vulnerabilities and patches
- Look for official documentation and security advisories
- Include compliance and regulatory considerations
- Search for industry best practices and benchmarks"""
        )
        
        return model
    
    async def search_and_respond(
        self, 
        query: str,
        include_citations: bool = True,
        search_params: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Process a query using Google Search grounding and return enhanced response.
        
        Args:
            query: User's query
            include_citations: Whether to include source citations
            search_params: Optional search parameters
            
        Returns:
            Dict containing response and metadata
        """
        try:
            # Configure generation parameters
            generation_config = GenerationConfig(
                temperature=0.7,
                top_p=0.95,
                max_output_tokens=2048,
            )
            
            # Configure safety settings
            safety_settings = [
                SafetySetting(
                    category=HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                    threshold=HarmBlockThreshold.BLOCK_ONLY_HIGH
                )
            ]
            
            # Generate response with Google Search grounding
            response = self.model.generate_content(
                query,
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            
            # Extract grounding metadata if available
            grounding_metadata = self._extract_grounding_metadata(response)
            
            # Format the response
            formatted_response = {
                "success": True,
                "query": query,
                "response": response.text,
                "grounding_metadata": grounding_metadata,
                "model_used": "gemini-1.5-pro with Google Search",
                "search_performed": True
            }
            
            # Add citations if requested and available
            if include_citations and grounding_metadata.get("sources"):
                formatted_response["citations"] = self._format_citations(
                    grounding_metadata["sources"]
                )
            
            logger.info(f"✅ Search-grounded response generated for: {query[:50]}...")
            return formatted_response
            
        except Exception as e:
            logger.error(f"❌ Error generating search-grounded response: {e}")
            return {
                "success": False,
                "query": query,
                "error": str(e),
                "response": "I encountered an error while searching for information. Please try again."
            }
    
    def _extract_grounding_metadata(self, response) -> Dict[str, Any]:
        """
        Extract grounding metadata from the model response.
        
        Args:
            response: GenerativeModel response
            
        Returns:
            Dict containing grounding metadata
        """
        metadata = {}
        
        try:
            # Check if response has grounding metadata
            if hasattr(response, 'grounding_metadata'):
                grounding = response.grounding_metadata
                
                # Extract search queries used
                if hasattr(grounding, 'search_queries'):
                    metadata["search_queries"] = grounding.search_queries
                
                # Extract sources/citations
                if hasattr(grounding, 'grounding_chunks'):
                    sources = []
                    for chunk in grounding.grounding_chunks:
                        source = {
                            "title": getattr(chunk.web, 'title', 'Unknown'),
                            "uri": getattr(chunk.web, 'uri', ''),
                        }
                        sources.append(source)
                    metadata["sources"] = sources
                
                # Extract retrieval confidence
                if hasattr(grounding, 'retrieval_confidence'):
                    metadata["confidence"] = grounding.retrieval_confidence
                    
        except Exception as e:
            logger.warning(f"Could not extract grounding metadata: {e}")
        
        return metadata
    
    def _format_citations(self, sources: List[Dict]) -> List[str]:
        """
        Format sources into readable citations.
        
        Args:
            sources: List of source dictionaries
            
        Returns:
            List of formatted citation strings
        """
        citations = []
        for i, source in enumerate(sources, 1):
            citation = f"[{i}] {source.get('title', 'Unknown')} - {source.get('uri', '')}"
            citations.append(citation)
        return citations
    
    async def search_security_topic(
        self,
        topic: str,
        include_latest: bool = True
    ) -> Dict[str, Any]:
        """
        Search for security-specific information with enhanced context.
        
        Args:
            topic: Security topic to search
            include_latest: Whether to specifically search for latest information
            
        Returns:
            Dict containing security-focused response
        """
        # Enhance query for security context
        enhanced_query = f"""Search for and provide comprehensive information about: {topic}

Please include:
1. Latest security vulnerabilities or issues related to {topic}
2. Current best practices and recommendations
3. Recent security advisories or updates (2024-2025)
4. GCP-specific security considerations
5. Compliance and regulatory aspects
6. Practical implementation steps

Focus on the most recent and authoritative sources."""
        
        if include_latest:
            enhanced_query += "\n\nPrioritize information from the last 6 months."
        
        return await self.search_and_respond(enhanced_query)
    
    async def analyze_with_search(
        self,
        query: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Analyze a query with optional context using Google Search enhancement.
        
        Args:
            query: Query to analyze
            context: Optional context dictionary
            
        Returns:
            Dict containing analysis with search-enhanced information
        """
        # Build contextual query
        contextual_query = query
        
        if context:
            contextual_query = f"""Given the following context:
{context}

Query: {query}

Please search for relevant information and provide a comprehensive response."""
        
        return await self.search_and_respond(contextual_query)


class ConversationalSearchAgent(SearchEnabledSecurityAgent):
    """Extended agent with conversational context and session management"""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        super().__init__(project_id, location)
        self.conversation_history = []
    
    async def search_with_context(
        self,
        query: str,
        session_id: Optional[str] = None,
        maintain_context: bool = True
    ) -> Dict[str, Any]:
        """
        Search with conversational context maintained.
        
        Args:
            query: User query
            session_id: Optional session identifier
            maintain_context: Whether to maintain conversation context
            
        Returns:
            Dict containing contextual response
        """
        # Add to conversation history
        if maintain_context:
            self.conversation_history.append({"role": "user", "content": query})
            
            # Include recent context in query
            if len(self.conversation_history) > 1:
                context_prompt = "Previous conversation:\n"
                for msg in self.conversation_history[-3:]:  # Last 3 messages
                    context_prompt += f"{msg['role']}: {msg['content'][:200]}...\n"
                
                query = f"{context_prompt}\n\nCurrent query: {query}"
        
        # Get search-enhanced response
        response = await self.search_and_respond(query)
        
        # Add response to history
        if maintain_context and response["success"]:
            self.conversation_history.append({
                "role": "assistant",
                "content": response["response"][:500]  # Store truncated version
            })
        
        # Add session metadata
        if session_id:
            response["session_id"] = session_id
            response["message_count"] = len(self.conversation_history)
        
        return response


# Factory function to create appropriate agent
def create_search_enabled_agent(
    project_id: str,
    agent_type: str = "basic",
    location: str = "us-central1"
) -> SearchEnabledSecurityAgent:
    """
    Factory function to create search-enabled agents.
    
    Args:
        project_id: GCP project ID
        agent_type: Type of agent ("basic" or "conversational")
        location: Vertex AI location
        
    Returns:
        Configured search-enabled agent
    """
    if agent_type == "conversational":
        return ConversationalSearchAgent(project_id, location)
    else:
        return SearchEnabledSecurityAgent(project_id, location)


# Example usage
if __name__ == "__main__":
    import asyncio
    
    async def test_search_agent():
        # Create agent
        agent = create_search_enabled_agent(
            project_id="your-project-id",
            agent_type="conversational"
        )
        
        # Test queries
        queries = [
            "What are the latest GCP security best practices?",
            "Search for recent vulnerabilities in container security",
            "Find information about zero trust architecture implementation"
        ]
        
        for query in queries:
            print(f"\n🔍 Query: {query}")
            response = await agent.search_and_respond(query)
            
            if response["success"]:
                print(f"✅ Response: {response['response'][:500]}...")
                if response.get("citations"):
                    print(f"📚 Sources: {response['citations']}")
            else:
                print(f"❌ Error: {response['error']}")
    
    # Run test
    asyncio.run(test_search_agent())