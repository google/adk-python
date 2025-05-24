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

import os
from app_container_framework.llm_agent import LLMAgent

class ResponderAgent(LLMAgent):
    """
    An agent that generates a response based on user input and a given category,
    using instructions loaded dynamically from a category-specific file.
    """

    def __init__(self, llm_instance=None):
        """
        Initializes the ResponderAgent.

        Args:
            llm_instance: An instance of an LLM to be used by the agent.
                          Defaults to None.
        """
        super().__init__(llm_instance=llm_instance)
        self.base_instructions_path = "contributing/samples/threat_intelligence_support/"

    def _load_instructions(self, category: str) -> str:
        """
        Loads instructions from a category-specific file.

        Args:
            category: The category name (e.g., "Malware Analysis").

        Returns:
            The instructions string.

        Raises:
            FileNotFoundError: If the instruction file for the category is not found.
        """
        # Normalize category name to filename (e.g., "Malware Analysis" -> "malware_analysis.txt")
        filename = category.lower().replace(" ", "_") + ".txt"
        filepath = os.path.join(self.base_instructions_path, filename)

        if not os.path.exists(filepath):
            # Fallback to a general guidelines file if specific category file doesn't exist
            filepath = os.path.join(self.base_instructions_path, "guidelines.txt")
            if not os.path.exists(filepath):
                raise FileNotFoundError(
                    f"No instruction file found for category '{category}' ({filename}) "
                    f"or default guidelines.txt at {self.base_instructions_path}"
                )

        with open(filepath, "r", encoding="utf-8") as f:
            return f.read().strip()

    def invoke(self, user_input: str, category: str) -> str:
        """
        Processes the user input and category to generate a response.

        Args:
            user_input: The user's input string (conversation thread).
            category: The classification category for the input.

        Returns:
            A string containing the generated response.
        """
        try:
            instructions = self._load_instructions(category)
        except FileNotFoundError as e:
            # Log the error and return a helpful message
            print(f"Error loading instructions: {e}") # Or use a proper logger
            return (
                "I am unable to process your request for this category "
                "due to missing configuration. Please contact support."
            )

        # Construct the prompt for the LLM
        prompt = f"{instructions}\n\nUser input: {user_input}\n\nResponse:"

        # Placeholder for actual LLM call
        # In a real scenario, you would pass this prompt to an LLM:
        # response_text = self.llm_instance.generate_text(prompt)
        # For this placeholder, we'll return a fixed response incorporating the category.
        response_text = (
            f"Placeholder response for category '{category}'. "
            f"Original input: '{user_input[:50]}...' "
            f"Instructions started with: '{instructions[:50]}...'"
        )

        return response_text

    # Optional: A run method if a different signature or simple passthrough is needed
    # For consistency with other agents, invoke is the primary method.
    def run(self, user_input: str, category: str) -> str:
        """
        Simple alias or wrapper for the invoke method.
        """
        return self.invoke(user_input, category)
