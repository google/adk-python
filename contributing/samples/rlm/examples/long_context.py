#!/usr/bin/env python3
"""
Long context example for ADK-RLM.

This example demonstrates how RLM handles long documents by
chunking and using recursive LLM calls.
"""

from adk_rlm import completion


def generate_long_document() -> str:
  """Generate a synthetic long document for testing."""
  sections = []

  topics = [
      (
          "Introduction to Machine Learning",
          "Machine learning is a subset of artificial intelligence...",
      ),
      (
          "Supervised Learning",
          "In supervised learning, models learn from labeled data...",
      ),
      (
          "Unsupervised Learning",
          "Unsupervised learning finds patterns in unlabeled data...",
      ),
      (
          "Deep Learning",
          "Deep learning uses neural networks with multiple layers...",
      ),
      (
          "Natural Language Processing",
          "NLP enables computers to understand human language...",
      ),
      (
          "Computer Vision",
          "Computer vision allows machines to interpret visual data...",
      ),
      (
          "Reinforcement Learning",
          "RL agents learn by interacting with environments...",
      ),
      (
          "Ethics in AI",
          "AI ethics considers fairness, transparency, and accountability...",
      ),
  ]

  for title, intro in topics:
    section = f"""
### {title}

{intro}

This section covers the fundamental concepts and applications of {title.lower()}.
The field has seen significant advances in recent years, with applications
ranging from healthcare to finance to entertainment.

Key concepts include:
- Theoretical foundations
- Practical implementations
- Real-world applications
- Current challenges and limitations
- Future directions

Many researchers and practitioners have contributed to this area,
making it one of the most active fields in computer science.
"""
    sections.append(section)

  return "\n\n".join(sections)


def main():
  # Generate a long document
  document = generate_long_document()
  print(f"Document length: {len(document)} characters")

  # Use completion with logging enabled
  result = completion(
      context=document,
      prompt=(
          "What are all the main topics covered in this document? List them"
          " with a brief summary of each."
      ),
      model="gemini-3-pro-preview",
      sub_model="gemini-3-flash-preview",
      max_iterations=15,
      log_dir="./logs",
      verbose=True,
  )

  print("\n" + "=" * 50)
  print("FINAL RESULT:")
  print("=" * 50)
  print(result.response)
  print(f"\nExecution time: {result.execution_time:.2f}s")


if __name__ == "__main__":
  main()
