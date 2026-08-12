from typing import Callable, Optional

from langchain_core.messages import ToolMessage

from src.agents.agents import build_search_agent, build_reader_agent, writer_chain, critic_chain


def _message_text(message) -> str:
    """Flatten a message's content (str, or Gemini's list-of-blocks) into plain text."""
    content = message.content
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            block["text"] if isinstance(block, dict) and block.get("type") == "text" else str(block)
            for block in content
        )
    return str(content)


def _raw_tool_output(messages) -> str:
    """Pull the tool calls' raw output (e.g. web_search's Title/URL/Snippet text)
    instead of the agent's paraphrased final answer, so links aren't lost."""
    return "\n----\n".join(_message_text(m) for m in messages if isinstance(m, ToolMessage))


def run_research_pipeline(topic: str, on_step: Optional[Callable[[str, str], None]] = None) -> dict:

    def notify(label: str, detail: str = "") -> None:
        if on_step:
            on_step(label, detail)

    state = {}

    #search agent working
    print("\n"+" ="*50)
    print("step 1 - search agent is working ...")
    print("="*50)
    notify("search", "Search agent is working ...")

    search_agent = build_search_agent()
    search_result = search_agent.invoke({
        "messages" : [("user", f"Find recent, reliable and detailed information about: {topic}")]
    })
    state["search_results"] = _raw_tool_output(search_result['messages']) or _message_text(search_result['messages'][-1])

    print("\n search result ",state['search_results'])
    notify("search_done", state["search_results"])


    #step 2 - reader agent
    print("\n"+" ="*50)
    print("step 2 - Reader agent is scraping top resources ...")
    print("="*50)
    notify("read", "Reader agent is scraping top resources ...")

    reader_agent = build_reader_agent()
    reader_result = reader_agent.invoke({
        "messages": [("user",
            f"Based on the following search results about '{topic}', "
            f"pick the most relevant URL and scrape it for deeper content.\n\n"
            f"Search Results:\n{state['search_results']}"
        )]
    })

    state['scraped_content'] = _raw_tool_output(reader_result['messages']) or _message_text(reader_result['messages'][-1])

    print("\nscraped content: \n", state['scraped_content'])
    notify("read_done", state["scraped_content"])


    #step 3 - writer chain

    print("\n"+" ="*50)
    print("step 3 - Writer is drafting the report ...")
    print("="*50)
    notify("write", "Writer is drafting the report ...")

    research_combined = (
        f"SEARCH RESULTS : \n {state['search_results']} \n\n"
        f"DETAILED SCRAPED CONTENT : \n {state['scraped_content']}"
    )

    state["report"] = writer_chain.invoke({
        "topic" : topic,
        "research" : research_combined
    })

    print("\n Final Report\n",state['report'])
    notify("write_done", state["report"])


    #critic report

    print("\n"+" ="*50)
    print("step 4 - critic is reviewing the report ")
    print("="*50)
    notify("critique", "Critic is reviewing the report ...")

    state["feedback"] = critic_chain.invoke({
        "report":state['report']
    })

    print("\n critic report \n", state['feedback'])
    notify("critique_done", state["feedback"])

    return state