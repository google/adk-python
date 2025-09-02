"""
Safe agent loading utility.
Handles complex import scenarios without directory switching.
"""
import sys
import os
import importlib.util
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

class AgentLoader:
    """Safe agent loader that doesn't require directory switching."""
    
    @staticmethod
    def load_agent(agent_name: str = "vertex_sqlite_agent", agent_dir: Optional[str] = None) -> Any:
        """
        Load an agent safely without directory switching.
        
        Args:
            agent_name: Name of the agent module (default: vertex_sqlite_agent)
            agent_dir: Optional path to agent directory
            
        Returns:
            Agent object with root_agent attribute
            
        Raises:
            ImportError: If agent cannot be loaded
        """
        if not agent_dir:
            agent_dir = AgentLoader._find_agent_directory()
        
        agent_path = Path(agent_dir)
        if not agent_path.exists():
            raise ImportError(f"Agent directory not found: {agent_path}")
        
        # Method 1: Try direct module loading
        try:
            return AgentLoader._load_with_importlib(agent_name, agent_path)
        except Exception as e:
            logger.warning(f"Method 1 failed: {e}")
        
        # Method 2: Try with sys.path manipulation (safer)
        try:
            return AgentLoader._load_with_syspath(agent_name, agent_path)
        except Exception as e:
            logger.warning(f"Method 2 failed: {e}")
        
        # Method 3: Try with directory context (last resort)
        try:
            return AgentLoader._load_with_context(agent_name, agent_path)
        except Exception as e:
            logger.error(f"Method 3 failed: {e}")
            raise ImportError(f"Failed to load agent {agent_name} from {agent_path}") from e
    
    @staticmethod
    def _find_agent_directory() -> Path:
        """Find the agent directory automatically."""
        # Start from this file and work up to find agents/gcp_security
        current = Path(__file__).resolve()
        
        # Look for agents/gcp_security relative to security_agent root
        for parent in current.parents:
            if parent.name == "security_agent":
                agent_dir = parent / "agents" / "gcp_security"
                if agent_dir.exists():
                    return agent_dir
        
        # Fallback: look relative to current file
        fallback = current.parent.parent / "agents" / "gcp_security"
        if fallback.exists():
            return fallback
        
        raise FileNotFoundError("Could not find agents/gcp_security directory")
    
    @staticmethod
    def _load_with_importlib(agent_name: str, agent_path: Path) -> Any:
        """Load agent using importlib.util (Method 1)."""
        module_file = agent_path / f"{agent_name}.py"
        if not module_file.exists():
            raise FileNotFoundError(f"Agent file not found: {module_file}")
        
        spec = importlib.util.spec_from_file_location(agent_name, module_file)
        if not spec or not spec.loader:
            raise ImportError(f"Could not create spec for {agent_name}")
        
        module = importlib.util.module_from_spec(spec)
        
        # Add agent directory to sys.path temporarily for imports
        original_path = sys.path.copy()
        try:
            if str(agent_path) not in sys.path:
                sys.path.insert(0, str(agent_path))
            
            spec.loader.exec_module(module)
            
            if hasattr(module, 'root_agent'):
                return module.root_agent
            elif hasattr(module, 'vertex_sqlite_agent'):
                return module.vertex_sqlite_agent
            else:
                raise AttributeError(f"No root_agent or vertex_sqlite_agent found in {agent_name}")
        
        finally:
            # Restore original sys.path
            sys.path = original_path
    
    @staticmethod
    def _load_with_syspath(agent_name: str, agent_path: Path) -> Any:
        """Load agent using sys.path manipulation (Method 2)."""
        original_path = sys.path.copy()
        
        try:
            # Add agent directory to path
            if str(agent_path) not in sys.path:
                sys.path.insert(0, str(agent_path))
            
            # Import the module
            if agent_name in sys.modules:
                # Force reload if already imported
                importlib.reload(sys.modules[agent_name])
                module = sys.modules[agent_name]
            else:
                module = importlib.import_module(agent_name)
            
            if hasattr(module, 'root_agent'):
                return module.root_agent
            elif hasattr(module, 'vertex_sqlite_agent'):
                return module.vertex_sqlite_agent
            else:
                raise AttributeError(f"No root_agent found in {agent_name}")
        
        finally:
            # Restore original sys.path
            sys.path = original_path
    
    @staticmethod
    def _load_with_context(agent_name: str, agent_path: Path) -> Any:
        """Load agent with directory context (Method 3 - last resort)."""
        original_cwd = Path.cwd()
        original_path = sys.path.copy()
        
        try:
            # Change to agent directory and add to path
            os.chdir(agent_path)
            if str(agent_path) not in sys.path:
                sys.path.insert(0, str(agent_path))
            
            # Import the module
            module = importlib.import_module(agent_name)
            
            if hasattr(module, 'root_agent'):
                return module.root_agent
            elif hasattr(module, 'vertex_sqlite_agent'):
                return module.vertex_sqlite_agent
            else:
                raise AttributeError(f"No root_agent found in {agent_name}")
        
        finally:
            # Always restore original state
            os.chdir(original_cwd)
            sys.path = original_path
    
    @staticmethod
    def get_agent_info(agent_dir: Optional[str] = None) -> dict:
        """
        Get information about the agent without loading it.
        
        Args:
            agent_dir: Optional path to agent directory
            
        Returns:
            dict: Agent information
        """
        try:
            if not agent_dir:
                agent_dir = AgentLoader._find_agent_directory()
            
            agent_path = Path(agent_dir)
            
            info = {
                "agent_directory": str(agent_path),
                "exists": agent_path.exists(),
                "files": [],
                "error": None
            }
            
            if agent_path.exists():
                # List Python files in the directory
                info["files"] = [f.name for f in agent_path.glob("*.py")]
                
                # Check for expected files
                expected_files = ["vertex_sqlite_agent.py", "sqlite_tool.py"]
                info["expected_files"] = {
                    filename: (agent_path / filename).exists()
                    for filename in expected_files
                }
            else:
                info["error"] = "Agent directory does not exist"
            
            return info
            
        except Exception as e:
            return {
                "agent_directory": str(agent_dir) if agent_dir else "unknown",
                "exists": False,
                "error": str(e),
                "files": []
            }

# Convenience function for backward compatibility
def load_agent() -> Any:
    """Load the default vertex_sqlite_agent."""
    return AgentLoader.load_agent()

def get_agent_info() -> dict:
    """Get agent information."""
    return AgentLoader.get_agent_info()