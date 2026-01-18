# Research Orchestration Agent

A multi-agent research pipeline demonstrating SequentialAgent orchestration
with Gemini models.

## Architecture

```
User Query → SearchAgent → ScraperAgent → CuratorAgent → WriterAgent → Report
```

## Features Demonstrated

* **SequentialAgent** - Pipeline orchestration pattern
* **Gemini 2.5 Flash** - Consistent model across all agents
* **Google Search grounding** - Built-in search tool
* **Custom tools** - Web scraping with BeautifulSoup

## Sample Query

* What are the latest developments in AI agent frameworks?
* Research the current state of autonomous agents.

## To Run

```bash
# CLI
adk run contributing/samples/research_orchestration

# Web UI
adk web contributing/samples
```

## Requirements

* `google-adk`: `pip install google-adk`
* BeautifulSoup: `pip install beautifulsoup4 lxml`
* Configure `GOOGLE_API_KEY` in `.env`
