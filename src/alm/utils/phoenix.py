import os
from openinference.instrumentation.langchain import LangChainInstrumentor
from phoenix.otel import register


def register_phoenix():
    # Get Phoenix endpoint from environment variable, defaults to localhost
    phoenix_endpoint = os.getenv("COLLECTOR_ENDPOINT")

    tracer_provider = register(
        project_name="ansible-log-monitor",
        endpoint=phoenix_endpoint,
        # auto_instrument=True,
    )

    # Explicitly instrument LangChain for Phoenix tracing
    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    return tracer_provider.get_tracer(__name__)
