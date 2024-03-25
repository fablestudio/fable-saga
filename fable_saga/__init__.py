import abc
import json
import logging
import pathlib
from typing import *

from langchain import LLMChain
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.llms.base import BaseLanguageModel
from langchain.prompts import load_prompt, BasePromptTemplate
from langchain.schema import LLMResult
from langchain.schema.output import Generation

# Package wide defaults.
default_openai_model_name = "gpt-3.5-turbo-1106"
default_openai_model_temperature = 0.9

# Set up logging.
logger = logging.getLogger(__name__)


class SagaCallbackHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

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

    def __init__(
        self,
        prompt_template: BasePromptTemplate,
        llm: Optional[BaseLanguageModel] = None,
    ):
        self._llm: BaseLanguageModel = (
            llm
            if llm is not None
            else ChatOpenAI(
                temperature=default_openai_model_temperature,
                model_kwargs={"response_format": {"type": "json_object"}},
            )
        )
        self.prompt_template = prompt_template

    def generate_chain(self, model_override: Optional[str] = None) -> LLMChain:
        if hasattr(self._llm, "model_name"):
            self._llm.model_name = (
                model_override if model_override else default_openai_model_name
            )
        return LLMChain(llm=self._llm, prompt=self.prompt_template)
