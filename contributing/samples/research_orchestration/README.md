# Research Orchestration Agent

A multi-agent research pipeline that combines **Gemini** and **DeepSeek** models to search, curate, and synthesize information from the internet.

## Architecture

```
User Query → SearchAgent (Gemini) → ScraperAgent (DeepSeek) → CuratorAgent (Gemini) → WriterAgent (DeepSeek) → Final Report
```

## Agent Configuration

| Agent | Model | Purpose |
|-------|-------|---------|
| SearchAgent | gemini-2.5-flash | Google Search grounding for finding sources |
| ScraperAgent | deepseek-chat | Extract content from web pages |
| CuratorAgent | gemini-2.5-flash | Filter and organize information |
| WriterAgent | deepseek-chat | Synthesize final report |

## Requirements

- `GOOGLE_API_KEY` - For Gemini models
- `DEEPSEEK_API_KEY` - For DeepSeek models
- LiteLLM installed (`pip install litellm`)
- BeautifulSoup installed (`pip install beautifulsoup4 lxml`)

## Usage

```bash
# CLI
adk run contributing/samples/research_orchestration

# Web UI
adk web contributing/samples
```

## Example Query

"What are the latest developments in AI agent frameworks in 2025?"
