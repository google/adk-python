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

import unittest
import sys
import os

# Add the parent directory of 'threat_intelligence_support' to sys.path
# to allow direct imports of agent modules. This assumes the test is run
# from the 'threat_intelligence_support' directory or a higher level that
# can resolve 'contributing.samples...'.

# Path adjustment to ensure modules can be found when tests are run.
# This goes up three levels from tests/ to the root of the ADK `app-container-framework-python`
# then appends the path to the agents.
current_dir = os.path.dirname(os.path.abspath(__file__)) # .../tests
sample_dir = os.path.dirname(current_dir) # .../threat_intelligence_support
# kit_dir = os.path.abspath(os.path.join(current_dir, "..", "..", "..", ".."))
# sys.path.insert(0, kit_dir)
sys.path.insert(0, os.path.abspath(os.path.join(sample_dir, "..", "..", "..")))


from contributing.samples.threat_intelligence_support.classifier_agent import ClassifierAgent
from contributing.samples.threat_intelligence_support.responder_agent import ResponderAgent
from app_container_framework.initialize import init_app_container_framework

class TestThreatIntelligenceAgents(unittest.TestCase):
    """
    Unit tests for the Threat Intelligence Support agents.
    """

    @classmethod
    def setUpClass(cls):
        """
        Set up for all tests in the class.
        Initialize the ADK framework.
        Ensure instruction files exist for the agents.
        """
        init_app_container_framework() # Initialize ADK

        # Base path where instruction files are located
        base_path = os.path.join(sample_dir, "") # Should point to threat_intelligence_support

        # Ensure classifier.txt exists
        classifier_txt_path = os.path.join(base_path, "classifier.txt")
        if not os.path.exists(classifier_txt_path):
            with open(classifier_txt_path, "w") as f:
                f.write("Instructions for the Classifier Agent: Analyze the user input and output a single category name.")

        # Ensure malware_analysis.txt exists
        malware_txt_path = os.path.join(base_path, "malware_analysis.txt")
        if not os.path.exists(malware_txt_path):
            with open(malware_txt_path, "w") as f:
                f.write("Instructions for responding to malware analysis requests.")

        # Ensure phishing_report.txt exists
        phishing_txt_path = os.path.join(base_path, "phishing_report.txt")
        if not os.path.exists(phishing_txt_path):
            with open(phishing_txt_path, "w") as f:
                f.write("Instructions for responding to phishing reports.")
        
        # Ensure guidelines.txt exists (for fallback)
        guidelines_txt_path = os.path.join(base_path, "guidelines.txt")
        if not os.path.exists(guidelines_txt_path):
            with open(guidelines_txt_path, "w") as f:
                f.write("General guidelines for Google Threat Intelligence Support agents.")


    def test_classifier_agent_instantiation_and_run(self):
        """Test that ClassifierAgent can be instantiated and its invoke method runs."""
        agent = ClassifierAgent(llm_instance=None)
        self.assertIsNotNone(agent, "ClassifierAgent should not be None after instantiation.")

        user_input = "My system is behaving strangely after I clicked a link."
        category = agent.invoke(user_input)

        self.assertIsInstance(category, str, "Category should be a string.")
        self.assertTrue(len(category) > 0, "Category string should not be empty.")
        # Based on current placeholder logic in ClassifierAgent:
        self.assertEqual(category, "General Inquiry", "Expected 'General Inquiry' for generic input.")

    def test_classifier_agent_malware_input(self):
        """Test ClassifierAgent with input suggesting malware."""
        agent = ClassifierAgent(llm_instance=None)
        user_input = "I found a suspicious file, I think it's malware."
        category = agent.invoke(user_input)
        self.assertEqual(category, "Malware Analysis", "Expected 'Malware Analysis' category.")

    def test_classifier_agent_phishing_input(self):
        """Test ClassifierAgent with input suggesting phishing."""
        agent = ClassifierAgent(llm_instance=None)
        user_input = "I got an email asking for my bank details, looks like phishing."
        category = agent.invoke(user_input)
        self.assertEqual(category, "Phishing Report", "Expected 'Phishing Report' category.")

    def test_responder_agent_instantiation_and_run(self):
        """Test that ResponderAgent can be instantiated and its invoke method runs."""
        agent = ResponderAgent(llm_instance=None)
        self.assertIsNotNone(agent, "ResponderAgent should not be None after instantiation.")

        user_input = "Can you analyze this file for me?"
        category = "Malware Analysis" # Test with a specific, valid category
        response = agent.invoke(user_input, category)

        self.assertIsInstance(response, str, "Response should be a string.")
        self.assertTrue(len(response) > 0, "Response string should not be empty.")
        self.assertIn(category, response, "Response should mention the category.")
        self.assertIn(user_input[:50], response, "Response should mention the user input.")

    def test_responder_agent_unknown_category_fallback(self):
        """Test ResponderAgent with an unknown category, expecting fallback."""
        agent = ResponderAgent(llm_instance=None)
        user_input = "What is the weather today?"
        category = "Weather Report" # An unknown category
        response = agent.invoke(user_input, category)

        self.assertIsInstance(response, str)
        self.assertTrue(len(response) > 0)
        # Expecting fallback to guidelines.txt
        self.assertIn("General guidelines", response, "Response should use general guidelines for unknown category.")

if __name__ == "__main__":
    unittest.main()
