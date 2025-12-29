import os

from langchain_openai import ChatOpenAI

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

# from langchain_core.globals import set_debug

# set_debug(True)  # Enables LangChain debug mode globally


def get_llm(model: str = MODEL, temperature: float = TEMPERATURE):
    llm = ChatOpenAI(
        api_key=API_KEY,
        base_url=BASE_URL,
        model=model,
        temperature=temperature,
    )
    return llm
