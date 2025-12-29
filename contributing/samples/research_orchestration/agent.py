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
Research Orchestration Agent

A multi-agent pipeline that searches the internet, curates information, 
and synthesizes reports using a mix of Gemini and DeepSeek models.

Model Configuration:
- SearchAgent: Gemini 2.5 Flash (for Google Search grounding)
- ScraperAgent: DeepSeek (for web content extraction)
- CuratorAgent: Gemini 2.5 Flash (for filtering and organizing)
- WriterAgent: DeepSeek (for final report synthesis)
"""

import requests
from bs4 import BeautifulSoup

from google.adk.agents.llm_agent import LlmAgent
from google.adk.agents.sequential_agent import SequentialAgent
from google.adk.tools import google_search
from google.adk.models.lite_llm import LiteLlm


# --- Custom Tool: Web Page Scraper ---
def load_web_page(url: str) -> str:
    """Fetches the content from a URL and returns the text.

    Args:
        url (str): The URL to fetch content from.

    Returns:
        str: The extracted text content from the web page.
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, 'lxml')
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            text = soup.get_text(separator='\n', strip=True)
            # Filter out very short lines
            lines = [line for line in text.splitlines() if len(line.split()) > 3]
            return '\n'.join(lines[:100])  # Limit to first 100 meaningful lines
        else:
            return f"Failed to fetch URL: {url} (Status: {response.status_code})"
    except Exception as e:
        return f"Error fetching URL {url}: {str(e)}"


# --- Model Configuration ---
# Gemini for search and curation (uses google_search grounding)
GEMINI_MODEL = "gemini-2.5-flash"

# DeepSeek for scraping and writing (via LiteLLM)
DEEPSEEK_MODEL = LiteLlm(model="deepseek/deepseek-chat")


# --- Agent 1: Search Agent (Gemini) ---
# Uses Google Search grounding to find relevant information
search_agent = LlmAgent(
    name="SearchAgent",
    model=GEMINI_MODEL,
    description="Searches the internet for relevant information using Google Search",
    instruction="""
    You are a research assistant specialized in finding information online.
    
    When given a research topic or question:
    1. Use the google_search tool to find relevant and recent information
    2. Identify the most credible and relevant sources
    3. Summarize what you found with key points
    4. List any important URLs that should be explored further
    
    Focus on finding factual, up-to-date information from reliable sources.
    """,
    tools=[google_search],
    output_key="search_results",
)


# --- Agent 2: Scraper Agent (DeepSeek) ---
# Extracts detailed content from web pages
scraper_agent = LlmAgent(
    name="ScraperAgent",
    model=DEEPSEEK_MODEL,
    description="Extracts and processes detailed content from web pages",
    instruction="""
    You are a web content extraction specialist.
    
    Based on the search results provided:
    {search_results}
    
    Your task:
    1. If specific URLs were mentioned in the search results, use the load_web_page tool to extract their content
    2. Process and clean the extracted content
    3. Identify the most valuable information from each source
    4. Compile all extracted information in an organized manner
    
    Focus on extracting substantive content, ignoring navigation, ads, and boilerplate text.
    """,
    tools=[load_web_page],
    output_key="scraped_content",
)


# --- Agent 3: Curator Agent (Gemini) ---
# Filters, organizes, and validates the gathered information
curator_agent = LlmAgent(
    name="CuratorAgent", 
    model=GEMINI_MODEL,
    description="Curates, filters, and organizes research information",
    instruction="""
    You are an expert information curator and fact-checker.
    
    Review all gathered information:
    - Search Results: {search_results}
    - Scraped Content: {scraped_content}
    
    Your task:
    1. Remove duplicate or redundant information
    2. Filter out low-quality, outdated, or irrelevant content
    3. Verify consistency across sources
    4. Organize information into logical themes or categories
    5. Identify key insights, patterns, and important findings
    6. Note any conflicting information or gaps in knowledge
    
    Output a well-organized summary of the curated information ready for report writing.
    """,
    output_key="curated_info",
)


# --- Agent 4: Writer Agent (DeepSeek) ---
# Synthesizes the final comprehensive report
writer_agent = LlmAgent(
    name="WriterAgent",
    model=DEEPSEEK_MODEL,
    description="Writes comprehensive reports from curated research",
    instruction="""
    You are an expert research report writer.
    
    Using the curated information provided:
    {curated_info}
    
    Write a comprehensive, well-structured research report that includes:
    
    1. **Executive Summary** - Brief overview of key findings (2-3 sentences)
    
    2. **Key Findings** - Main discoveries organized by theme, supported by evidence
    
    3. **Analysis** - Your interpretation of what the findings mean
    
    4. **Recommendations** (if applicable) - Actionable insights based on the research
    
    5. **Sources** - List the key sources that informed this report
    
    Write in a clear, professional tone. Use bullet points for clarity where appropriate.
    Ensure the report is informative, accurate, and actionable.
    """,
    output_key="final_report",
)


# --- Main Pipeline: Sequential Orchestration ---
# Chains the agents: Search → Scrape → Curate → Write
root_agent = SequentialAgent(
    name="ResearchOrchestration",
    sub_agents=[
        search_agent,    # Step 1: Find sources (Gemini + Google Search)
        scraper_agent,   # Step 2: Extract content (DeepSeek)
        curator_agent,   # Step 3: Organize & filter (Gemini)
        writer_agent,    # Step 4: Write report (DeepSeek)
    ],
    description=(
        "A research pipeline that searches the internet, extracts content, "
        "curates information, and produces a comprehensive report. "
        "Uses Gemini for search/curation and DeepSeek for extraction/writing."
    ),
)
