# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from app_container_framework.llm_agent import LLMAgent

class ClassifierAgent(LLMAgent):
    """
    An agent that classifies user input based on instructions
    loaded from a file.
    """

    def __init__(self, llm_instance=None):
        """
        Initializes the ClassifierAgent.

        Args:
            llm_instance: An instance of an LLM to be used by the agent.
                          Defaults to None.
        """
        super().__init__(llm_instance=llm_instance)
        self.instructions = self._load_instructions()

    def _load_instructions(self) -> str:
        """Loads instructions from the classifier.txt file."""
        with open("contributing/samples/threat_intelligence_support/classifier.txt", "r", encoding="utf-8") as f:
            return f.read().strip()

    def run(self, user_input: str) -> str:
        """
        Processes the user input and returns a classification category.

        Args:
            user_input: The user's input string (conversation thread).

        Returns:
            A string representing the classification category.
        """
        # Placeholder for actual LLM call.
        # For now, it uses the loaded instructions and user input
        # to make a simple determination, or returns a fixed category.

        prompt = f"{self.instructions}\n\nUser input: {user_input}\n\nCategory:"

        # In a real scenario, you would pass this prompt to an LLM.
        # For this placeholder, we'll return a fixed category.
        # A more sophisticated placeholder might derive the category
        # from the input, e.g., if "malware" in user_input.lower(): return "Malware Analysis"
        if "malware" in user_input.lower():
            return "Malware Analysis"
        elif "phishing" in user_input.lower():
            return "Phishing Report"
        else:
            return "General Inquiry" # Default fixed category

    # The invoke method is often used for more complex interactions,
    # but for this simple agent, `run` is sufficient.
    # If an invoke method is preferred by the ADK for all agents,
    # it can be added here, potentially calling the run method.
    def invoke(self, user_input: str) -> str:
        """
        Invokes the agent with user input.

        Args:
            user_input: The user's input string.

        Returns:
            The result of the agent's processing, which is a category string.
        """
        return self.run(user_input)
