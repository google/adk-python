# audio_agent.py

from google.adk.agents.llm_agent import LlmAgent
import json
from pathlib import Path

LOG_FILE = Path("logs/audio_log.json")

class AudioAgent(LlmAgent):
    """
    An agent that receives a live audio stream and transcribes it to text
    using a specialized Gemini streaming model.
    """
    # **BUG FIX:** Changed the agent name to be a valid Python identifier.
    name: str = "Audio_Agent"
    model: str = "gemini-2.0-flash-live-preview-04-09"
    prompt: str = "You are a live transcriptionist. Transcribe the user's speech accurately."

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
        if is_final and response:
            try:
                with open(LOG_FILE, "a") as f:
                    json.dump({"transcription": response}, f)
                    f.write("\n")
            except Exception as e:
                print(f"Error writing to audio log file: {e}")