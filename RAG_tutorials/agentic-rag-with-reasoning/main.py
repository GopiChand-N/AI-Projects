import sys
from config import get_openai_key
from knowledge_loader import load_knowledge
from agent_initializer import load_agent


def main():
    try:
        openai_key = get_openai_key()
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Load the knowledge base using the OpenAI API key for embeddings
    print("Loading knowledge base...")
    knowledge = load_knowledge(openai_key)

    # Initialize the agent using the OpenAI API key for the language model
    print("Loading agent...")
    agent = load_agent(knowledge, openai_key)

    # Display current knowledge sources
    print("\nCurrent knowledge sources:")
    for i, url in enumerate(knowledge.urls, start=1):
        print(f"{i}. {url}")

    # Optionally allow the user to add a new knowledge source URL
    add_url = input("\nWould you like to add a new URL to the knowledge base? (y/n): ").strip().lower()
    if add_url == 'y':
        new_url = input("Enter new URL: ").strip()
        if new_url:
            print("Loading new documents...")
            knowledge.urls.append(new_url)
            # Load new documents without recreating the entire database
            knowledge.load(recreate=False, upsert=True, skip_existing=True)
            print(f"Added: {new_url}")

    # Get the user query
    query = input("\nEnter your question: ").strip()
    if not query:
        print("No question provided. Exiting.")
        return

    print("\nProcessing your query...\n")
    answer_text = ""
    reasoning_text = ""
    citations = []

    try:
        for chunk in agent.run(query, stream=True, show_full_reasoning=True, stream_intermediate_steps=True):
            # Write intermediate reasoning steps to the console
            if chunk.reasoning_content:
                reasoning_text = chunk.reasoning_content
                print("Reasoning step:", reasoning_text)

            # Append and display answer chunks
            if chunk.content and chunk.event in {"run_response", "run_completed"}:
                if isinstance(chunk.content, str):
                    answer_text += chunk.content
                    print("Answer:", answer_text)

            # Collect citations
            if chunk.citations and chunk.citations.urls:
                citations = chunk.citations.urls
    except Exception as e:
        print("Error occurred during agent execution:", e)

    # Display citations if available
    if citations:
        print("\nSources:")
        for cite in citations:
            title = cite.title if cite.title else cite.url
            print(f"- {title}: {cite.url}")


if __name__ == "__main__":
    main()