import os

import streamlit as st
from dotenv import load_dotenv

from src.pipelines.pipeline import run_research_pipeline

load_dotenv()

st.set_page_config(page_title="Multi-Agent Research Assistant", page_icon="🔎")

st.title("🔎 Multi-Agent Research Assistant")
st.caption(
    "Search agent → reader agent → writer → critic. "
    "Runs on Gemini's free tier, rate-limited to about one request every "
    "2 seconds, so a full run usually takes 1-2 minutes."
)

REQUIRED_KEYS = ["GEMINI_API_KEY", "TAVILY_API_KEY"]

STEP_LABELS = {
    "search": "🔍 Searching the web ...",
    "read": "📖 Scraping the top result ...",
    "write": "✍️ Drafting the report ...",
    "critique": "🧐 Critiquing the report ...",
}

topic = st.text_input(
    "Research topic",
    placeholder="e.g. The impact of AI on the job market in 2026",
)

if st.button("Run Research", type="primary"):
    missing_keys = [key for key in REQUIRED_KEYS if not os.getenv(key)]

    if not topic.strip():
        st.error("Enter a topic first.")
    elif missing_keys:
        st.error(f"Missing required API key(s) in .env: {', '.join(missing_keys)}")
    else:
        with st.status("Running research pipeline ...", expanded=True) as status:

            def on_step(label: str, detail: str) -> None:
                if label in STEP_LABELS:
                    status.update(label=STEP_LABELS[label])
                elif detail:
                    preview = detail[:500] + ("..." if len(detail) > 500 else "")
                    status.write(preview)

            state = run_research_pipeline(topic, on_step=on_step)
            status.update(label="Done", state="complete")

        st.session_state["result"] = {"topic": topic, **state}

result = st.session_state.get("result")

if result:
    st.subheader(f"Results for: {result['topic']}")

    report_tab, feedback_tab, search_tab, scrape_tab = st.tabs(
        ["Report", "Critic Feedback", "Search Results", "Scraped Content"]
    )

    with report_tab:
        st.markdown(result["report"])
        st.download_button(
            "Download report (.md)",
            data=result["report"],
            file_name="research_report.md",
            mime="text/markdown",
        )

    with feedback_tab:
        st.markdown(result["feedback"])

    with search_tab:
        st.text(result["search_results"])

    with scrape_tab:
        st.text(result["scraped_content"])
