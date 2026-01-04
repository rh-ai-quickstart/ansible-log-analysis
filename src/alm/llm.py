import os
from typing import List, Dict
from langchain_openai import ChatOpenAI

from alm.utils.logger import get_logger

logger = get_logger(__name__)

# Constants for API configuration
API_KEY: str = os.getenv("OPENAI_API_TOKEN")
BASE_URL: str = os.getenv("OPENAI_API_ENDPOINT")

# Constants for model configuration
MODEL: str = os.getenv("OPENAI_MODEL")
TEMPERATURE: float = float(os.getenv("OPENAI_TEMPERATURE"))

if not API_KEY or not BASE_URL:
    raise ValueError(
        "OpenAI API configuration not found. Please set both OPENAI_API_TOKEN and OPENAI_API_ENDPOINT environment variables."
    )

# from langchain.globals import set_debug

# set_debug(True)  # Enables LangChain debug mode globally


def get_llm(model: str = MODEL, temperature: float = TEMPERATURE):
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=model,
        temperature=temperature,
    )
    return llm


def get_streaming_llm(model: str = MODEL, temperature: float = TEMPERATURE):
    """Returns an LLM configured for streaming responses."""
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=model,
        temperature=temperature,
        streaming=True,
    )
    return llm


async def stream_with_fallback(llm, messages: List[Dict[str, str]]):
    """
    Asynchronously stream response and collect chunks.
    Returns whatever was received even if an error occurs mid-stream.
    """
    collected_output = []

    try:
        async for chunk in llm.astream(messages):
            if chunk.content:
                collected_output.append(chunk.content)
    except Exception as e:
        # Log the error but continue with what we have
        logger.error(f"Stream interrupted: {e}")
        if len(collected_output) == 0:
            raise e
    return "".join(collected_output)
