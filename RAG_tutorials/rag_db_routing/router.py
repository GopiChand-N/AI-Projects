from agno.agent import Agent
from agno.models.openai import OpenAIChat
import streamlit as st
from typing import Optional, Literal
from config import COLLECTIONS

DatabaseType = Literal["products", "support", "finance"]


def create_routing_agent() -> Agent:
    """Creates a routing agent using agno framework"""
    return Agent(
        model=OpenAIChat(
            id="gpt-4o",
            api_key=st.session_state.openai_api_key
        ),
        tools=[],
        description="""You are a query routing expert. Your only job is to analyze questions and determine which database they should be routed to.
        You must respond with exactly one of these three options: 'products', 'support', or 'finance'. The user's question is: {question}""",
        instructions=[
            "Follow these rules strictly:",
            "1. For questions about products, features, specifications, or item details, or product manuals → return 'products'",
            "2. For questions about help, guidance, troubleshooting, or customer service, FAQ, or guides → return 'support'",
            "3. For questions about costs, revenue, pricing, or financial data, or financial reports and investments → return 'finance'",
            "4. Return ONLY the database name, no other text or explanation",
            "5. If you're not confident about the routing, return an empty response"
        ],
        markdown=False,
        show_tool_calls=False
    )


def route_query(question: str) -> Optional[DatabaseType]:
    """Route query by searching all databases and comparing relevance scores.
    Returns None if no suitable database is found."""
    try:
        best_score = -1
        best_db_type = None
        all_scores = {}  # Store all scores for debugging

        # Search each database and compare relevance scores
        for db_type, db in st.session_state.databases.items():
            results = db.similarity_search_with_score(
                question,
                k=3
            )

            if results:
                avg_score = sum(score for _, score in results) / len(results)
                all_scores[db_type] = avg_score

                if avg_score > best_score:
                    best_score = avg_score
                    best_db_type = db_type

        confidence_threshold = 0.5
        if best_score >= confidence_threshold and best_db_type:
            st.success(f"Using vector similarity routing: {best_db_type} (confidence: {best_score:.3f})")
            return best_db_type

        st.warning(f"Low confidence scores (below {confidence_threshold}), falling back to LLM routing")

        # Fallback to LLM routing
        routing_agent = create_routing_agent()
        response = routing_agent.run(question)

        db_type = (response.content
                   .strip()
                   .lower()
                   .translate(str.maketrans('', '', '`\'"')))

        if db_type in COLLECTIONS:
            st.success(f"Using LLM routing decision: {db_type}")
            return db_type

        st.warning("No suitable database found, will use web search fallback")
        return None

    except Exception as e:
        st.error(f"Routing error: {str(e)}")
        return None