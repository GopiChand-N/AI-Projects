from agno.agent import Agent
from agno.models.openai import OpenAIChat
from agno.tools.reasoning import ReasoningTools


def load_agent(knowledge, openai_key):
    """
    Create and return an agent with reasoning capabilities using OpenAI.

    Parameters:
        knowledge: The knowledge base object.
        openai_key (str): OpenAI API key.

    Returns:
        Agent: An agent instance ready to process queries.
    """
    agent = Agent(
        model=OpenAIChat(
            id = "gpt-4.1-mini",
            api_key=openai_key
        ),
        knowledge=knowledge,
        search_knowledge=True,  # Enable knowledge search
        tools=[ReasoningTools(add_instructions=True)],  # Add reasoning tools
        instructions=[
            "Include sources in your response.",
            "Always search your knowledge before answering the question.",
        ],
        markdown=True  # Enable markdown formatting if needed
    )
    return agent