# camera_agent.py

from google.adk.agents.llm_agent import LlmAgent
import json
from pathlib import Path

LOG_FILE = Path("logs/video_log.json")

class CameraAgent(LlmAgent):
    """
    An agent that receives video frames, interprets them using a Gemini model,
    and logs the interpretation.
    """
    # **BUG FIX:** Changed the agent name to be a valid Python identifier.
    name: str = "Camera_Agent"
    model: str = "gemini-2.0-flash"
    prompt: str = "You are a helpful assistant. Describe what you see in this image in one short sentence."

    def on_agent_response(
        self,
        response: str,
        *,
        is_tool_code: bool = False,
        is_final: bool = False,
        turn_id: str,
        session_id: str,
        run_config_id: str,
    ) -> None:
        super().on_agent_response(
            response,
            is_tool_code=is_tool_code,
            is_final=is_final,
            turn_id=turn_id,
            session_id=session_id,
            run_config_id=run_config_id,
        )
        if is_final:
            try:
                with open(LOG_FILE, "a") as f:
                    json.dump({"interpretation": response}, f)
                    f.write("\n")
            except Exception as e:
                print(f"Error writing to video log file: {e}")