"""Test sequential parallel agents to verify common prefix visibility."""

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.parallel_agent import ParallelAgent
from google.adk.agents.sequential_agent import SequentialAgent
from tests.unittests import testing_utils


def test_sequential_parallels():
    """Test Sequential[Parallel1[A,B], Parallel2[D,E]].
    
    D and E should be able to see A and B's outputs because:
    - Parallel1 creates: "Parallel1.A", "Parallel1.B"
    - Parallel1 joins: ctx.branch = "Parallel1"
    - Parallel2 creates: "Parallel1.Parallel2.D", "Parallel1.Parallel2.E"
    - Common prefix check: "Parallel1.Parallel2.D" and "Parallel1.A" share "Parallel1"
    """
    # Parallel1 agents
    alice_model = testing_utils.MockModel.create(responses=["I am Alice"])
    alice = LlmAgent(
        name="Alice",
        description="Agent A",
        instruction="Say: I am Alice",
        model=alice_model,
    )
    
    bob_model = testing_utils.MockModel.create(responses=["I am Bob"])
    bob = LlmAgent(
        name="Bob",
        description="Agent B",
        instruction="Say: I am Bob",
        model=bob_model,
    )
    
    # Parallel2 agents - David should see Alice and Bob
    david_model = testing_utils.MockModel.create(responses=["I am David"])
    david = LlmAgent(
        name="David",
        description="Agent D",
        instruction="Respond based on context",
        model=david_model,
    )
    
    eve_model = testing_utils.MockModel.create(responses=["I am Eve"])
    eve = LlmAgent(
        name="Eve",
        description="Agent E",
        instruction="Respond based on context",
        model=eve_model,
    )
    
    # Create parallel groups
    parallel1 = ParallelAgent(
        name="Parallel1",
        description="First parallel group",
        sub_agents=[alice, bob],
    )
    
    parallel2 = ParallelAgent(
        name="Parallel2",
        description="Second parallel group",
        sub_agents=[david, eve],
    )
    
    # Create sequential agent
    root = SequentialAgent(
        name="Root",
        description="Sequential of parallels",
        sub_agents=[parallel1, parallel2],
    )
    
    # Run the agent
    runner = testing_utils.InMemoryRunner(root_agent=root)
    runner.run("Start")
    session = runner.session
    
    # Print branch contexts for debugging
    print("\n=== Branch Hierarchy ===")
    for event in session.events:
        if event.author and event.branch:
            print(f"{event.author:15} | branch={event.branch}")
    
    # Helper to extract text from simplified contents
    def extract_text(contents):
        texts = []
        for role, content in contents:
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if hasattr(part, 'text') and part.text:
                        texts.append(part.text)
            elif hasattr(content, 'text') and content.text:
                texts.append(content.text)
        return " ".join(texts)
    
    # David (in Parallel2) should see Alice and Bob from Parallel1
    assert len(david_model.requests) > 0, "David should have made LLM requests"
    david_contents = testing_utils.simplify_contents(david_model.requests[0].contents)
    david_text = extract_text(david_contents)
    
    print(f"\nDavid's LLM request text (first 300 chars):\n{david_text[:300]}")
    
    assert "Alice" in david_text or "I am Alice" in david_text, \
        f"David should see Alice's output. Got: {david_text[:200]}"
    assert "Bob" in david_text or "I am Bob" in david_text, \
        f"David should see Bob's output. Got: {david_text[:200]}"
    
    print("\n✅ SUCCESS! David can see Alice and Bob (common prefix filtering works!)")


if __name__ == "__main__":
    test_sequential_parallels()
