import abc
import logging
from typing import *

from langchain.chains import LLMChain
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import BasePromptTemplate
from langchain.schema import LLMResult
from langchain.schema.output import Generation
from langchain_openai import ChatOpenAI

# Package wide defaults.
default_openai_model_name = "gpt-3.5-turbo-1106"
default_openai_model_temperature = 0.9

# Set up logging.
logger = logging.getLogger(__name__)


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

    async def on_llm_start(
        self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        """Run when LLM starts running."""
        self.last_prompt = prompts[-1] if len(prompts) > 0 else None
        if self.prompt_callback is not None:
            self.prompt_callback(prompts)

    async def on_llm_end(self, response: LLMResult, **kwargs: Any) -> Any:
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

        self._llm: BaseLanguageModel = (
            llm
            if llm is not None
            # Use the default OpenAI model if no model is provided.
            else ChatOpenAI(
                temperature=default_openai_model_temperature,
                # Set the response format to JSON object, this feature is specific to a subset of OpenAI models,
                # but it seems to help a lot as we expect a JSON object response.
                model_kwargs={"response_format": {"type": "json_object"}},
                callbacks=[self.callback_handler],
            )
        )
        self.prompt_template = prompt_template

    def generate_chain(self, model_override: Optional[str] = None) -> LLMChain:
        """Generate an LLMChain for the agent. (see LangChain docs)."""
        if hasattr(self._llm, "model_name"):
            # If this model has a model_name attribute, set it to the override if provided. Useful to change models at
            # runtime, for instance when using OpenAI and allowing the caller to specify which specific model to use
            # per request.
            self._llm.model_name = (
                model_override if model_override else default_openai_model_name
            )
        return LLMChain(llm=self._llm, prompt=self.prompt_template)
