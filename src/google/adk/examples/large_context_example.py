# Copyright 2025 Google LLC
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

"""
Large Context Example - Demonstrating Efficient Context Management

This example demonstrates how to use the context reference store and large context state
to efficiently handle very large context windows (1M-2M tokens) with ADK and Gemini.
"""

import time
import sys
import json
import random
from typing import Dict, Any, List

from google.adk.agents import LlmAgent
from google.adk.sessions import LargeContextState, ContextReferenceStore
from google.adk.tools import FunctionTool


def generate_large_document(token_size: int = 500000) -> Dict[str, Any]:
    """
    Generate a mock document of approximately specified token size with realistic structure.

    Args:
        token_size: Approximate size in tokens

    Returns:
        A structured document
    """
    # 1 token ≈ 4 characters in English
    char_count = token_size * 4

    # Create sample paragraphs to simulate document content
    paragraphs = [
        "This is a sample document with information that might be relevant to the query.",
        "It contains multiple paragraphs with different content to simulate a real document.",
        "Some paragraphs are longer and contain more detailed information about specific topics.",
        "Others are shorter and provide concise summaries or transitions between sections.",
        "The document also includes sections with headings, lists, and other structured content.",
        "This helps simulate the complexity of real documents processed by AI systems.",
        "Technical documents often contain specialized terminology and references.",
        "Research papers include citations, methodology descriptions, and analysis of results.",
        "Legal documents have specific formatting, defined terms, and complex nested clauses.",
        "Narrative text might include dialogue, character descriptions, and scene settings.",
    ]

    # Create a complex structure to simulate real documents
    document = {
        "metadata": {
            "title": "Large Context Processing Example Document",
            "created_at": "2024-11-14T10:30:00Z",
            "author": "ADK Context Management Example",
            "version": "1.0",
            "tags": ["example", "large_context", "gemini", "adk"],
            "summary": "A synthetic document for demonstrating large context processing",
        },
        "sections": [],
    }

    # Generate enough sections and content to reach the desired token size
    current_char_count = len(json.dumps(document))
    section_id = 1

    while current_char_count < char_count:
        # Create a section with subsections
        section = {
            "id": f"section-{section_id}",
            "title": f"Section {section_id}: {random.choice(['Overview', 'Analysis', 'Results', 'Discussion', 'Methods'])}",
            "content": "\n\n".join(random.sample(paragraphs, min(5, len(paragraphs)))),
            "subsections": [],
        }

        # Add subsections
        for j in range(1, random.randint(3, 6)):
            subsection = {
                "id": f"section-{section_id}-{j}",
                "title": f"Subsection {section_id}.{j}",
                "content": "\n\n".join(
                    random.sample(paragraphs, min(3, len(paragraphs)))
                ),
                "paragraphs": [],
            }

            # Add more detailed paragraphs to subsections
            for k in range(1, random.randint(5, 15)):
                paragraph = {
                    "id": f"para-{section_id}-{j}-{k}",
                    "text": random.choice(paragraphs),
                    "metadata": {
                        "relevance_score": round(random.random(), 2),
                        "keywords": random.sample(
                            [
                                "ai",
                                "context",
                                "processing",
                                "testing",
                                "gemini",
                                "adk",
                                "efficiency",
                            ],
                            3,
                        ),
                    },
                }
                subsection["paragraphs"].append(paragraph)

            section["subsections"].append(subsection)

        document["sections"].append(section)
        current_char_count = len(json.dumps(document))
        section_id += 1

        # Safety check to avoid infinite loops
        if section_id > 1000:
            break

    print(
        f"Generated document with approximate size: {current_char_count / 4:.0f} tokens"
    )
    return document


def extract_section(
    context_state: LargeContextState, section_id: str
) -> Dict[str, Any]:
    """
    Extract a specific section from the document by ID.

    Args:
        context_state: State with document reference
        section_id: ID of the section to extract

    Returns:
        The extracted section
    """
    # Retrieve the document from the context store
    document = context_state.get_context("document_ref")

    # Search for the section with the given ID
    for section in document.get("sections", []):
        if section.get("id") == section_id:
            return section

        # Check subsections if not found at top level
        for subsection in section.get("subsections", []):
            if subsection.get("id") == section_id:
                return subsection

    return {"error": f"Section with ID {section_id} not found"}


def search_document(
    context_state: LargeContextState, keywords: List[str]
) -> List[Dict[str, Any]]:
    """
    Search for keywords in the document and return matching paragraphs.

    Args:
        context_state: State with document reference
        keywords: List of keywords to search for

    Returns:
        List of matching paragraphs with metadata
    """
    # Retrieve the document from the context store
    document = context_state.get_context("document_ref")

    # Normalize keywords for case-insensitive search
    normalized_keywords = [k.lower() for k in keywords]

    # Search for matches
    matches = []

    for section in document.get("sections", []):
        # Check section content
        section_content = section.get("content", "").lower()
        if any(k in section_content for k in normalized_keywords):
            matches.append(
                {
                    "id": section.get("id"),
                    "title": section.get("title"),
                    "match_type": "section",
                    "content_preview": (
                        section.get("content")[:200] + "..."
                        if len(section.get("content", "")) > 200
                        else section.get("content")
                    ),
                }
            )

        # Check subsections
        for subsection in section.get("subsections", []):
            subsection_content = subsection.get("content", "").lower()
            if any(k in subsection_content for k in normalized_keywords):
                matches.append(
                    {
                        "id": subsection.get("id"),
                        "title": subsection.get("title"),
                        "match_type": "subsection",
                        "content_preview": (
                            subsection.get("content")[:200] + "..."
                            if len(subsection.get("content", "")) > 200
                            else subsection.get("content")
                        ),
                    }
                )

            # Check paragraphs
            for paragraph in subsection.get("paragraphs", []):
                paragraph_text = paragraph.get("text", "").lower()
                if any(k in paragraph_text for k in normalized_keywords):
                    matches.append(
                        {
                            "id": paragraph.get("id"),
                            "match_type": "paragraph",
                            "text": paragraph.get("text"),
                            "relevance_score": paragraph.get("metadata", {}).get(
                                "relevance_score"
                            ),
                        }
                    )

    return matches


def run_example():
    """Run the large context example."""
    print("Starting Large Context Example...")

    # Create a context store
    context_store = ContextReferenceStore()

    # Create a large context state
    state = LargeContextState(context_store=context_store)

    # Generate a large document
    print("Generating large test document...")
    start_time = time.time()
    document = generate_large_document(token_size=500000)
    generation_time = time.time() - start_time
    print(f"Document generation time: {generation_time:.2f} seconds")

    # Store the document in the context store
    print("Storing document in context store...")
    start_time = time.time()
    document_ref = state.add_large_context(
        document,
        metadata={"content_type": "application/json", "cache_ttl": 3600},
        key="document_ref",
    )
    store_time = time.time() - start_time
    print(f"Document storage time: {store_time:.2f} seconds")
    print(f"Document reference ID: {document_ref}")

    # Retrieve the document from the context store
    print("Retrieving document from context store...")
    start_time = time.time()
    retrieved_document = state.get_context("document_ref")
    retrieval_time = time.time() - start_time
    print(f"Document retrieval time: {retrieval_time:.2f} seconds")

    # Create function tools that use the context
    extract_section_tool = FunctionTool(
        func=extract_section,
        name="extract_section",
        description="Extract a specific section from the document by ID",
    )

    search_document_tool = FunctionTool(
        func=search_document,
        name="search_document",
        description="Search for keywords in the document and return matching paragraphs",
    )

    # Create an agent that uses the tools
    agent = LlmAgent(
        name="document_explorer",
        model="gemini-1.5-pro-latest",
        instruction="""
        You are a document explorer agent. You have access to a large document 
        through reference-based context management. You can:
        
        1. Extract specific sections by ID using the extract_section tool
        2. Search for keywords in the document using the search_document tool
        
        Always use these tools to access the document rather than trying to 
        process the entire document at once.
        """,
        tools=[extract_section_tool, search_document_tool],
        description="Agent for exploring large documents efficiently",
    )

    print("\nAgent created with tools for exploring the document.")
    print("This example demonstrates how to use the context reference store")
    print("to efficiently manage large contexts with ADK and Gemini.")
    print(
        "\nIn a real application, you would use the agent to interact with the document."
    )
    print("For example, you could call:")
    print('  agent.run({"user_input": "Find sections about AI", "state": state})')

    # For this example, I will just demonstrate searching for a keyword
    print("\nDemonstrating document search...")
    search_results = search_document(state, ["ai", "context"])
    print(f"Found {len(search_results)} matches for 'ai' or 'context'")

    # Print a few results
    for i, result in enumerate(search_results[:3]):
        print(f"\nMatch {i+1}:")
        print(f"  ID: {result.get('id')}")
        print(f"  Type: {result.get('match_type')}")
        if "title" in result:
            print(f"  Title: {result.get('title')}")
        if "text" in result:
            print(f"  Text: {result.get('text')}")
        if "content_preview" in result:
            print(f"  Preview: {result.get('content_preview')}")

    if len(search_results) > 3:
        print(f"\n... and {len(search_results) - 3} more matches.")


if __name__ == "__main__":
    run_example()
