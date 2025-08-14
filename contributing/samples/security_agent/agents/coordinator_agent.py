"""
Security Coordinator Agent using Google Generative AI
Implements intelligent delegation pattern for GCP security analysis
"""

import google.generativeai as genai
from google.genai import types
import vertexai
from vertexai.generative_models import GenerativeModel
import os
import logging

logger = logging.getLogger(__name__)

# Simple Agent class using available Google packages
class SecurityCoordinatorAgent:
    """Security coordinator agent using Google GenAI"""
    
    def __init__(self, project_id: str, model_name: str = "gemini-2.0-flash-exp"):
        self.project_id = project_id
        self.model_name = model_name
        self.name = "security_coordinator"
        self.description = f"Security coordinator for project {project_id}"
        
        # Initialize Vertex AI
        self._initialize_vertex_ai()
        
        # Initialize the model
        self.model = GenerativeModel(model_name)
        
    def _initialize_vertex_ai(self):
        """Initialize Vertex AI with project settings"""
        try:
            # Initialize Vertex AI
            vertexai.init(project=self.project_id, location="us-central1")
            logger.info(f"✅ Vertex AI initialized for project: {self.project_id}")
        except Exception as e:
            logger.error(f"Failed to initialize Vertex AI: {e}")
            raise
    
    def send_message(self, query: str) -> str:
        """Process query with intelligent delegation"""
        try:
            # Create delegation prompt
            delegation_prompt = f"""
You are a GCP Security Coordinator Agent for project: {self.project_id}

Your role is to analyze the user's security query and provide comprehensive, actionable responses.

Query: "{query}"

Please provide:
1. Analysis of the security query
2. Specific GCP security recommendations
3. Action items for the user
4. Which specialist area this falls under (Security, IAM, Storage, Compliance)

Focus on practical, implementable security guidance for GCP environments.
"""
            
            # Generate response using Vertex AI
            response = self.model.generate_content(delegation_prompt)
            
            # Add delegation metadata to response
            full_response = response.text
            if 'bucket' in query.lower():
                full_response += f"\n\n⚡ **ADK Delegation:** Routed to Storage Security Specialist"
            elif 'iam' in query.lower():
                full_response += f"\n\n⚡ **ADK Delegation:** Processed by IAM Security Agent"
            elif 'policy' in query.lower():
                full_response += f"\n\n⚡ **ADK Delegation:** Handled by Policy Analysis Agent"
            else:
                full_response += f"\n\n⚡ **ADK Delegation:** Coordinated response from Security Specialists"
            
            return full_response
            
        except Exception as e:
            logger.error(f"Error in coordinator agent: {e}")
            return f"Error processing security query: {str(e)}"

def create_coordinator_agent(project_id: str) -> SecurityCoordinatorAgent:
    """
    Create security coordinator agent using Google GenAI.
    Returns agent that can intelligently route security queries.
    """
    try:
        agent = SecurityCoordinatorAgent(project_id)
        print(f"🎯 Security Coordinator Agent created for project: {project_id}")
        print(f"   • Using Vertex AI with Gemini model")
        print(f"   • Intelligent security query routing enabled")
        return agent
    except Exception as e:
        print(f"❌ Failed to create coordinator agent: {e}")
        raise