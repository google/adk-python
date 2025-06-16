# LinkedIn Post Generator Loop Agent

This example demonstrates the use of a Sequential and Loop Agent pattern in the Agent Development Kit (ADK) to generate and refine a LinkedIn post.

## Overview

The LinkedIn Post Generator uses a sequential pipeline with a loop component to:
1. Generate an initial LinkedIn post
2. Iteratively refine the post until quality requirements are met

This demonstrates several key patterns:
1. **Sequential Pipeline**: A multi-step workflow with distinct stages
2. **Iterative Refinement**: Using a loop to repeatedly refine content
3. **Automatic Quality Checking**: Validating content against specific criteria
4. **Feedback-Driven Refinement**: Improving content based on specific feedback
5. **Loop Exit Tool**: Using a tool to terminate the loop when quality requirements are met

## Architecture

The system is composed of the following components:

### Root Sequential Agent

`LinkedInPostGenerationPipeline` - A SequentialAgent that orchestrates the overall process:
1. First runs the initial post generator
2. Then executes the refinement loop

### Initial Post Generator

`InitialPostGenerator` - An LlmAgent that creates the first draft of the LinkedIn post with no prior context.

### Refinement Loop

`PostRefinementLoop` - A LoopAgent that executes a two-stage refinement process:
1. First runs the reviewer to evaluate the post and possibly exit the loop
2. Then runs the refiner to improve the post if the loop continues

### Sub-Agents Inside the Refinement Loop

1. **Post Reviewer** (`PostReviewer`) - Reviews posts for quality and provides feedback or exits the loop if requirements are met
2. **Post Refiner** (`PostRefiner`) - Refines the post based on feedback to improve quality

### Tools

1. **Character Counter** - Validates post length against requirements (used by the Reviewer)
2. **Exit Loop** - Terminates the loop when all quality criteria are satisfied (used by the Reviewer)

## Loop Control with Exit Tool

A key design pattern in this example is the use of an `exit_loop` tool to control when the loop terminates. The Post Reviewer has two responsibilities:

1. **Quality Evaluation**: Checks if the post meets all requirements
2. **Loop Control**: Calls the exit_loop tool when the post passes all quality checks

When the exit_loop tool is called:
1. It sets `tool_context.actions.escalate = True`
2. This signals to the LoopAgent that it should stop iterating

This approach follows ADK best practices by:
1. Separating initial generation from refinement
2. Giving the quality reviewer direct control over loop termination
3. Using a dedicated agent for post refinement
4. Using a tool to manage the loop control flow

## Usage

To run this example:

```bash
cd 11-loop-agent
adk web
```

Then in the web interface, enter a prompt like:
"Generate a LinkedIn post about what I've learned from @aiwithbrandon's Agent Development Kit tutorial."

The system will:
1. Generate an initial LinkedIn post
2. Review the post for quality and compliance with requirements
3. If the post meets all requirements, exit the loop
4. Otherwise, provide feedback and refine the post
5. Continue this process until a satisfactory post is created or max iterations reached
6. Return the final post

## Example Input

```
Generate a LinkedIn post about what I've learned from @aiwithbrandon's Agent Development Kit tutorial.
```

## Loop Termination

The loop terminates in one of two ways:
1. When the post meets all quality requirements (reviewer calls the exit_loop tool)
2. After reaching the maximum number of iterations (10)
