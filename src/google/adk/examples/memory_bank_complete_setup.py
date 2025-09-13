"""
Complete Memory Bank Service setup example - Issue #2940 Solution

This example demonstrates the correct way to:
1. Create an Agent Engine
2. Extract the Agent ID from the resource name
3. Setup Memory Bank Service with the correct ID

Common Error: Using api_resource.name directly instead of extracting the ID
Solution: Use .split("/")[-1] to extract the ID from the full resource path
"""

import os
import sys
import logging
import vertexai
from google.adk.memory import VertexAiMemoryBankService
from google.adk.sessions import VertexAiSessionService
from google.adk.agents import Agent
from google.adk.tools.preload_memory_tool import PreloadMemoryTool
import google.adk as adk

# Import our utility function
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from google.adk.utils.resource_utils import extract_agent_engine_id, validate_agent_engine_resource_name

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MemoryBankSetup:
    """Complete Memory Bank Service setup with proper agent ID extraction."""
    
    def __init__(self, project_id: str, location: str = "us-central1"):
        self.project_id = project_id
        self.location = location
        self.agent_engine = None
        self.agent_engine_id = None
        self.memory_service = None
        self.session_service = None
        
    def create_agent_engine(self):
        """Create Vertex AI Agent Engine and extract ID correctly."""
        try:
            # Create Vertex AI client
            client = vertexai.Client(project=self.project_id, location=self.location)
            
            # Create Agent Engine
            self.agent_engine = client.agent_engines.create()
            resource_name = self.agent_engine.api_resource.name
            
            logger.info(f"Created Agent Engine: {resource_name}")
            
            # CRITICAL: Extract Agent ID from resource name (Issue #2940 fix)
            if validate_agent_engine_resource_name(resource_name):
                self.agent_engine_id = extract_agent_engine_id(resource_name)
                logger.info(f"Extracted Agent Engine ID: {self.agent_engine_id}")
            else:
                raise ValueError(f"Invalid Agent Engine resource name format: {resource_name}")
                
            return self.agent_engine_id
            
        except Exception as e:
            logger.error(f"Failed to create Agent Engine: {str(e)}")
            raise
    
    def setup_memory_service(self):
        """Setup Memory Bank Service with correct agent ID."""
        if not self.agent_engine_id:
            raise ValueError("Agent Engine ID not available. Call create_agent_engine() first.")
        
        try:
            # Create Memory Bank Service using extracted agent ID
            self.memory_service = VertexAiMemoryBankService(
                project=self.project_id,
                location=self.location,
                agent_engine_id=self.agent_engine_id  # Use extracted ID, NOT full resource name
            )
            
            logger.info("Memory Bank Service setup complete")
            return self.memory_service
            
        except Exception as e:
            logger.error(f"Failed to setup Memory Bank Service: {str(e)}")
            raise
    
    def setup_session_service(self):
        """Setup Session Service with correct agent ID."""
        if not self.agent_engine_id:
            raise ValueError("Agent Engine ID not available. Call create_agent_engine() first.")
        
        try:
            # Create Session Service using extracted agent ID
            self.session_service = VertexAiSessionService(
                project_id=self.project_id,
                location=self.location,
                agent_engine_id=self.agent_engine_id  # Use extracted ID
            )
            
            logger.info("Session Service setup complete")
            return self.session_service
            
        except Exception as e:
            logger.error(f"Failed to setup Session Service: {str(e)}")
            raise
    
    def create_agent_with_memory(self):
        """Create an agent with memory capabilities."""
        if not self.memory_service or not self.session_service:
            raise ValueError("Memory and Session services must be setup first")
        
        try:
            # Create agent with memory tools
            agent = adk.Agent(
                model="gemini-2.5-flash",
                name="memory-enabled-agent",
                instruction="You are an AI assistant with persistent memory. Remember user preferences and conversation history.",
                description="Agent with Memory Bank capabilities for persistent conversations",
                tools=[PreloadMemoryTool()]
            )
            
            # Create runner with memory and session services
            runner = adk.Runner(
                agent=agent,
                app_name="memory-bank-demo-app",
                session_service=self.session_service,
                memory_service=self.memory_service
            )
            
            logger.info("Agent with Memory Bank created successfully")
            return runner
            
        except Exception as e:
            logger.error(f"Failed to create agent with memory: {str(e)}")
            raise

def main():
    """Main function demonstrating complete setup."""
    # Configuration
    project_id = os.getenv("GOOGLE_CLOUD_PROJECT")
    location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    
    if not project_id:
        print("Error: GOOGLE_CLOUD_PROJECT environment variable not set")
        print("Set it with: export GOOGLE_CLOUD_PROJECT='your-project-id'")
        return
    
    print(f"Starting Memory Bank Service setup for project: {project_id}")
    print(f"Location: {location}")
    print("-" * 60)
    
    try:
        # Initialize setup class
        setup = MemoryBankSetup(project_id=project_id, location=location)
        
        # Step 1: Create Agent Engine and extract ID
        print("Creating Agent Engine...")
        agent_id = setup.create_agent_engine()
        print(f"   Agent Engine ID: {agent_id}")
        
        # Step 2: Setup Memory Bank Service
        print("Setting up Memory Bank Service...")
        setup.setup_memory_service()
        
        # Step 3: Setup Session Service
        print("Setting up Session Service...")
        setup.setup_session_service()
        
        # Step 4: Create Agent with Memory
        print("Creating Agent with Memory...")
        runner = setup.create_agent_with_memory()
        
        print("-" * 60)
        print("SUCCESS! Memory Bank Service setup complete!")
        print(f"Agent Engine ID: {agent_id}")
        print("Agent is ready for conversations with persistent memory")
        print("-" * 60)
        
        # Optional: Test the setup
        print("Testing memory capabilities...")
        response = runner.run("Hello! Please remember that my favorite color is blue.")
        print(f"Agent: {response}")
        
    except Exception as e:
        print(f"Setup failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
