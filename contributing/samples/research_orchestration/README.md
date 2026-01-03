# Research Orchestration Agent

A multi-agent research pipeline demonstrating SequentialAgent orchestration
with multi-model support via LiteLLM.

## Architecture

```
User Query → SearchAgent → ScraperAgent → CuratorAgent → WriterAgent → Report
```

## Features Demonstrated

* **SequentialAgent** - Pipeline orchestration pattern
* **LiteLLM integration** - Multi-model support
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

* `google-adk` with LiteLLM extension: `pip install google-adk[extensions]`
* BeautifulSoup: `pip install beautifulsoup4 lxml`
* Configure API keys in `.env` for your chosen models
