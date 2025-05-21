from google.adk.agents.sequential_agent import SequentialAgent

from .introduction_agent import IntroductionAgent
from .needs_assessment_agent import NeedsAssessmentAgent
from .financial_pre_approval_agent import FinancialPreApprovalAgent
from .property_preferences_agent import PropertyPreferencesAgent
from .closing_agent import ClosingAgent

# Instantiate the sub-agents
introduction_agent_instance = IntroductionAgent()
needs_assessment_agent_instance = NeedsAssessmentAgent()
financial_pre_approval_agent_instance = FinancialPreApprovalAgent()
property_preferences_agent_instance = PropertyPreferencesAgent()
closing_agent_instance = ClosingAgent()

# Create the SequentialAgent instance
HomebuyerInterviewAgent = SequentialAgent(
    name="HomebuyerInterviewAgent",
    sub_agents=[
        introduction_agent_instance,
        needs_assessment_agent_instance,
        financial_pre_approval_agent_instance,
        property_preferences_agent_instance,
        closing_agent_instance,
    ],
    description="A sequential agent that conducts a comprehensive interview with a new homebuyer.",
)

# Assign the HomebuyerInterviewAgent instance to root_agent
root_agent = HomebuyerInterviewAgent
