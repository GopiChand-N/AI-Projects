import os
from dotenv import load_dotenv

import agno.knowledge
print(dir(agno.knowledge.knowledge.Reader))
# Load environment variables from a .env file
load_dotenv()

def get_openai_key():
    """
    Retrieve the OpenAI API key from environment variables.
    Returns:
        str: The OpenAI API key.
    Raises:
        ValueError: if the API key is missing.
    """
    openai_key = os.getenv("OPENAI_API_KEY")
    if not openai_key:
        raise ValueError("Missing API key. Please set OPENAI_API_KEY in your environment.")
    return openai_key