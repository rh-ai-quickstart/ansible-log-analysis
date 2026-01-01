import os

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


def get_llm_support_tool_calling():
    """
    Get LLM instance specifically for agents that support tool calling.
    As Workaround, uses litemaas endpoint which supports tool calling, while other agents use the default RHOAI model.
    """
    API_KEY_WITH_TOOL_CALLING: str = os.getenv("OPENAI_API_TOKEN_WITH_TOOL_CALLING")
    BASE_URL_WITH_TOOL_CALLING: str = os.getenv("OPENAI_API_ENDPOINT_WITH_TOOL_CALLING")
    MODEL_WITH_TOOL_CALLING: str = os.getenv("OPENAI_MODEL_WITH_TOOL_CALLING")

    if (
        API_KEY_WITH_TOOL_CALLING
        and BASE_URL_WITH_TOOL_CALLING
        and MODEL_WITH_TOOL_CALLING
    ):
        return ChatOpenAI(
            api_key=API_KEY_WITH_TOOL_CALLING,
            base_url=BASE_URL_WITH_TOOL_CALLING,
            model=MODEL_WITH_TOOL_CALLING,
            temperature=TEMPERATURE,
        )
    else:
        logger.warning(
            "OpenAI API configuration for tool calling not found. Trying default RHOAI model."
        )
        return get_llm()
