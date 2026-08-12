# LangChain Multi-Agent Research System

A multi-agent research assistant built with [LangChain](https://www.langchain.com/) and [Google Gemini](https://ai.google.dev/). Give it a topic, and a pipeline of cooperating agents searches the web, reads the most relevant source, drafts a structured report, and critiques its own output — all exposed through a simple Streamlit UI.

## How it works

The pipeline runs four stages in sequence, each handled by a dedicated agent or chain:

```
 topic
   │
   ▼
┌─────────────────┐      ┌─────────────────┐      ┌───────────────┐      ┌────────────────┐
│  Search Agent    │ ──▶ │  Reader Agent    │ ──▶ │  Writer Chain  │ ──▶ │  Critic Chain   │
│  (web_search)     │     │  (scrape_url)    │     │  (LLM)         │     │  (LLM)          │
└─────────────────┘      └─────────────────┘      └───────────────┘      └────────────────┘
        │                        │                        │                       │
        ▼                        ▼                        ▼                       ▼
  titles/URLs/snippets   cleaned article text     structured report        score + feedback
  from Tavily search      scraped from the top       (intro, findings,
                           result                       sources)
```

1. **Search Agent** — a LangChain tool-calling agent that queries the [Tavily](https://tavily.com/) search API and returns titles, URLs, and snippets.
2. **Reader Agent** — picks the most relevant URL from the search results and scrapes it for deeper content, using `trafilatura`/`readability`/`BeautifulSoup` as fallback extraction strategies.
3. **Writer Chain** — a prompt → LLM → output-parser chain that turns the gathered research into a structured report (introduction, key findings, conclusion, sources).
4. **Critic Chain** — reviews the report and returns a score, strengths, and areas to improve.

All LLM calls go through a single `gemini-flash-lite-latest` model instance, client-side rate-limited to stay within Gemini's free-tier quota.

## Tech stack

- **[LangChain](https://www.langchain.com/)** (`langchain`, `langchain-core`, `langchain-community`) — agent orchestration, prompt templates, tool calling
- **[langchain-google-genai](https://python.langchain.com/docs/integrations/chat/google_generative_ai/)** — Gemini chat model integration
- **[Tavily](https://tavily.com/)** — web search API
- **[Streamlit](https://streamlit.io/)** — interactive web UI
- **trafilatura / readability-lxml / BeautifulSoup4** — layered HTML content extraction for web scraping
- **python-dotenv** — environment variable management
- **rich** — pretty console logging

## Project structure

```
.
├── app.py                     # standalone script to try the search/scrape tools directly
├── main.py                    # standalone script to run the pipeline from the terminal
├── streamlit_app.py           # Streamlit UI entrypoint
├── requirements.txt
└── src/
    ├── agents/
    │   └── agents.py          # Gemini model setup + search agent, reader agent, writer/critic chains
    ├── pipelines/
    │   └── pipeline.py        # orchestrates the 4-stage pipeline, step-by-step
    └── tools/
        └── tools.py           # web_search (Tavily) and scrape_url tools
```

## Installation

**Prerequisites:** Python 3.11+, and API keys for [Tavily](https://tavily.com/) and [Google AI Studio](https://aistudio.google.com/app/apikey) (Gemini).

1. Clone the repository
   ```bash
   git clone https://github.com/<your-username>/LangChain-Multi-Agent-Research-System.git
   cd LangChain-Multi-Agent-Research-System
   ```

2. Create and activate a virtual environment
   ```bash
   conda create -n langagent python=3.11 -y
   conda activate langagent
   ```
   or, with `venv`:
   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. Install dependencies
   ```bash
   pip install -r requirements.txt
   ```

4. Configure environment variables — create a `.env` file in the project root:
   ```env
   GEMINI_API_KEY=your_gemini_api_key
   TAVILY_API_KEY=your_tavily_api_key
   ```

## Usage

**Streamlit UI** (recommended):
```bash
streamlit run streamlit_app.py
```
Enter a research topic and click **Run Research**. A full run takes roughly 1-2 minutes due to Gemini's free-tier rate limit. Results are shown across four tabs: Report, Critic Feedback, Search Results, and Scraped Content, with a markdown download button for the final report.

**Command line:**
```bash
python main.py
```
Edit the `topic` variable in `main.py` to change what gets researched.

## License

Distributed under the Apache License 2.0. See [LICENSE](LICENSE) for details.
