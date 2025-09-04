from langgraph.prebuilt import create_react_agent
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.language_models import BaseLanguageModel
from langchain.schema import HumanMessage
import streamlit as st


def create_fallback_agent(chat_model: BaseLanguageModel):
    """Create a LangGraph agent for web research."""

    def web_research(query: str) -> str:
        """Web search with result formatting."""
        try:
            search = DuckDuckGoSearchRun(num_results=5)
            results = search.run(query)
            return results
        except Exception as e:
            return f"Search failed: {str(e)}. Providing answer based on general knowledge."

    tools = [web_research]

    agent = create_react_agent(model=chat_model,
                               tools=tools,
                               debug=False)

    return agent


def _handle_web_fallback(question: str) -> tuple[str, list]:
    st.info("No relevant documents found. Searching web...")
    fallback_agent = create_fallback_agent(st.session_state.llm)

    with st.spinner('Researching...'):
        agent_input = {
            "messages": [
                HumanMessage(content=f"Research and provide a detailed answer for: '{question}'")
            ],
            "is_last_step": False
        }

        try:
            response = fallback_agent.invoke(agent_input, config={"recursion_limit": 100})
            if isinstance(response, dict) and "messages" in response:
                answer = response["messages"][-1].content
                return f"Web Search Result:\n{answer}", []

        except Exception:
            # Fallback to general LLM response
            fallback_response = st.session_state.llm.invoke(question).content
            return f"Web search unavailable. General response: {fallback_response}", []