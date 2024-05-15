import abc
import logging
from typing import *

from langchain.chains import LLMChain
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import BasePromptTemplate
from langchain.schema import LLMResult
from langchain.schema.output import Generation

# Set up logging.
logger = logging.getLogger(__name__)
streaming_debug_logger = logging.getLogger(__name__ + ".streaming_debug")


class StreamingDebugCallback(AsyncCallbackHandler):
    """Callback handler that prints the response as it comes in."""

    def __init__(self):
        self.response: str = ""
        self.last_token: str = ""

    def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ):
        """Run on LLM start."""
        # Reset the response and last good response.
        self.response = ""

        if streaming_debug_logger.isEnabledFor(logging.INFO):
            print("\n-> Generating ..", flush=True)

    def on_llm_end(self, response: LLMResult, **kwargs):
        """Run on LLM end."""
        if streaming_debug_logger.isEnabledFor(logging.INFO):
            print(
                "\n-> Done!",
                flush=True,
            )

    def on_llm_new_token(self, token: str, **kwargs):
        """Run on new LLM token. Only available when streaming is enabled."""
        self.response += token
        # The json mode of ollama (mistra:instruct at least) sends a lot of newlines at the end of the response.
        # We don't want to print them.
        if token == "\n" and self.last_token == "\n":
            return
        if streaming_debug_logger.level == logging.DEBUG:
            # If we are in debug mode, print everything (words).
            print(token, end="", flush=True)
        elif streaming_debug_logger.level == logging.INFO:
            # If we are in info mode, print only dots to indicate progress.
            print(".", end="", flush=True)
        self.last_token = token


class SagaCallbackHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain. (see LangChain docs)."""

    def __init__(
        self,
        prompt_callback: Optional[Callable[[List[str]], None]] = None,
        response_callback: Optional[Callable[[LLMResult], None]] = None,
    ):
        super().__init__()
        self.last_prompt: Optional[str] = None
        self.last_generation: Optional[str] = None
        self.last_model_info: Optional[dict[str, Any]] = None
        self.prompt_callback = prompt_callback
        self.response_callback = response_callback

    def on_llm_new_token(self, token: str, **kwargs):
        if self.last_model_info is not None:
            # If we have model info, we can calculate the token usage.
            # See below for the on_llm_start method where we set the last_model_info
            # to get around the issue of OpenAI not returning the token count in the response.
            self.last_model_info["token_usage"]["completion_tokens"] += 1

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.last_prompt = prompts[-1] if len(prompts) > 0 else None
        if self.prompt_callback is not None:
            self.prompt_callback(prompts)

        # If streaming is enabled, currently OpenAI doesn't return the count of tokens in the response.
        # so we need to create the llm_info ourselves.
        # See https://github.com/langchain-ai/langchain/issues/13430
        params = kwargs.get("invocation_params", {})
        if (
            "stream" in params
            and params["stream"]
            and "_type" in params
            and params["_type"] == "openai-chat"
        ):
            # Use the tiktoken library to get the token counts.
            import tiktoken

            model_name = params.get("model_name")
            enc = tiktoken.encoding_for_model(model_name)
            token_count = 0
            # For each prompt, count the tokens.
            for prompt in prompts:
                tokens = enc.encode(prompt)
                token_count += len(tokens)
            # Set the last model info, so it can be used later to count completion tokens and then
            # returned in the response.
            self.last_model_info = {
                "model_name": params.get("model_name"),
                "token_usage": {"prompt_tokens": token_count, "completion_tokens": 0},
            }

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
        # Don't overwrite the last model info if it's already set (e.g. when OpenAI streaming above).
        if self.last_model_info is None:
            self.last_model_info = response.llm_output
        if len(response.generations) > 0:
            # TODO: This is a hack to get the last generation. We should fix this in langchain.
            flat_generation = response.generations[0][0]
            if isinstance(flat_generation, Generation):
                self.last_generation = flat_generation.text
        if self.response_callback is not None:
            self.response_callback(response)


class BaseSagaAgent(abc.ABC):
    """Base class for SAGA agents."""

    def __init__(
        self,
        prompt_template: BasePromptTemplate,
        llm: Optional[BaseLanguageModel] = None,
    ):
        """Initialize the agent.

        Args:
            prompt_template: The prompt template to use for generation (see LangChain docs).
            llm: The language model to use for generation, if None, ChatOpenAI is used. (see LangChain docs).
        """
        # Set up the callback handler.
        self.callback_handler = SagaCallbackHandler(
            prompt_callback=lambda prompts: logger.info(f"Prompts: {prompts}"),
            response_callback=lambda result: logger.info(f"Response: {result}"),
        )

        if llm is not None:
            assert isinstance(
                llm, BaseLanguageModel
            ), "llm must inherit from BaseLanguageModel."
            self._llm = llm
        else:
            logger.warning("No language model provided, using default OpenAI model.")
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "langchain-openai not found. Please install langchain-openai (e.g `poetry install --extras openai`) or provide a specific llm."
                )
            self._llm = ChatOpenAI(
                temperature=0.9,
                model="gpt-3.5-turbo",
                # Set the response format to JSON object, this feature is specific to a subset of OpenAI models,
                # but it seems to help a lot as we expect a JSON object response.
                model_kwargs={"response_format": {"type": "json_object"}},
                callbacks=[self.callback_handler, StreamingDebugCallback()],
                streaming=True,
            )

        self.prompt_template = prompt_template

    def generate_chain(self) -> LLMChain:
        """Generate an LLMChain for the agent. (see LangChain docs)."""
        return LLMChain(llm=self._llm, prompt=self.prompt_template)
