# Homebuyer Interview Agent

This agent is designed to conduct a comprehensive interview with a potential new homebuyer. It guides the user through several stages, gathering information about their needs, preferences, and financial situation.

## Purpose

The primary goal of this agent is to collect all necessary information from a homebuyer to help them find a suitable property. It follows a structured interview process, ensuring all key aspects are covered.

## Structure

The agent is implemented as a `SequentialAgent` composed of several sub-agents, each responsible for a specific stage of the interview:

1.  **IntroductionAgent**: Welcomes the homebuyer and asks initial questions.
2.  **NeedsAssessmentAgent**: Gathers information about the homebuyer's property requirements (type of home, size, location, must-have features, etc.).
3.  **FinancialPreApprovalAgent**: Discusses the buyer's budget, pre-approval status, and other financial considerations.
4.  **PropertyPreferencesAgent**: Inquires about preferences for the property search process, lifestyle needs, and desired amenities.
5.  **ClosingAgent**: Summarizes the gathered information, outlines the next steps, and answers any final questions.

## How to Run

To run this agent locally, navigate to the `contributing/samples/homebuyer_interview_agent/` directory in your terminal and run:

```bash
python main.py
```

This will start an interactive session with the agent in your terminal.
