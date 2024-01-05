import json
import logging
from typing import *
import pathlib

import cattrs
from attr import define
from cattr import unstructure, structure
from langchain import LLMChain
from langchain.callbacks.base import AsyncCallbackHandler
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import load_prompt
from langchain.schema import LLMResult
from langchain.schema.output import Generation

# Package wide defaults.
default_openai_model_name = "gpt-3.5-turbo-1106"
default_openai_model_temperature = 0.9

# Set up logging.
logger = logging.getLogger(__name__)


@define(slots=True)
class Skill:
    name: str
    description: str
    parameters: Dict[str, str] = {}


@define(slots=True)
class Action:
    skill: str
    parameters: Dict[str, Any] = {}


@define(slots=True)
class GeneratedActions:
    options: List[Action]
    scores: List[float]
    raw_prompt: Optional[str] = None
    raw_response: Optional[str] = None
    llm_info: Optional[Dict[str, Any]] = None
    retries: int = 0
    error: Optional[str] = None

    def sort(self):
        """Sort the actions by score."""
        self.options = [x for _, x in sorted(zip(self.scores, self.options), key=lambda pair: pair[0], reverse=True)]
        self.scores = sorted(self.scores, reverse=True)


class SagaCallbackHandler(AsyncCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    def __init__(self, prompt_callback: Optional[Callable[[List[str]], None]] = None,
                 response_callback: Optional[Callable[[LLMResult], None]] = None):
        super().__init__()
        self.last_prompt: Optional[str] = None
        self.last_generation: Optional[str] = None
        self.last_model_info: Optional[str] = None
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
            #TODO: This is a hack to get the last generation. We should fix this in langchain.
            flat_generation = response.generations[0][0]
            if isinstance(flat_generation, Generation):
                self.last_generation = flat_generation.text
        if self.response_callback is not None:
            self.response_callback(response)


class Agent:
    """SAGA Agent """

    def __init__(self, llm: BaseChatModel = None):
        self._llm = llm if llm is not None else \
            ChatOpenAI(temperature=default_openai_model_temperature, model_name=default_openai_model_name,
                       model_kwargs={
                           "response_format": {"type": "json_object"}})
        self.guidance = """
* Provide a goal for your character to achieve as well as the action.
* Do not make up new characters.
* Staying true to your character's personality and circumstances.
* Use your observations and conversations to inform your actions.
* Only use provided skills to create action options (do not make up skills).
* Do not do the same thing again if you have already done it.
* Advance the story by generating actions that will help you achieve your most important and immediate goals.
"""
        path = pathlib.Path(__file__).parent.resolve()
        self.prompt = load_prompt(path / "prompt_templates/generate_actions.yaml")

    def chain(self, model_override: Optional[str] = None) -> LLMChain:
        self._llm.model_name = model_override if model_override else default_openai_model_name
        return LLMChain(llm=self._llm, prompt=self.prompt)

    async def generate_actions(self, context: str, skills: List[Skill], max_tries=0, verbose=False, model_override: Optional[str] = None) -> GeneratedActions:
        """Generate actions for the given context and skills."""
        assert context is not None and len(context) > 0, "Must provide a context."
        assert skills is not None and len(skills) > 0, "Must provide at least one skill."
        for skill in skills:
            assert isinstance(skill, Skill), "Must provide a list of Skill objects."
            assert skill.name is not None and len(skill.name) > 0, "Must provide a skill name."
            assert skill.description is not None and len(skill.description) > 0, "Must provide a skill description."

        chain = self.chain(model_override)
        chain.verbose = verbose

        # Set up the callback handler.
        callback_handler = SagaCallbackHandler(
            prompt_callback=lambda prompts: logger.info(f"Prompts: {prompts}"),
            response_callback=lambda result: logger.info(f"Response: {result}")
        )

        retries = 0
        last_error = None
        dumped_skills = [unstructure(skill) for skill in skills]
        json_skills = json.dumps(dumped_skills)

        while retries <= max_tries:
            try:
                last_response = await chain.arun(context=context, skills=json_skills, callbacks=[callback_handler])
                raw_actions = json.loads(last_response)
                # If we parse the results, but didn't get any options, retry. Should be rare.
                if len(raw_actions.get('options', [])) == 0:
                    print("No options found. Retrying.")
                    retries += 1
                    continue

                # Convert the options to a GeneratedActions object and add metadata.
                actions = structure(raw_actions, GeneratedActions)
                actions.raw_prompt = callback_handler.last_prompt
                actions.raw_response = last_response
                actions.llm_info = callback_handler.last_model_info
                actions.retries = retries

                # Sort the actions by score to be helpful before returning.
                actions.sort()
                return actions
            except (json.JSONDecodeError, TypeError) as e:
                last_error = f"Error decoding response: {str(e)}"
            except cattrs.errors.ClassValidationError as e:
                last_error = f"Error validating response: {cattrs.transform_error(e)}"
            except Exception as e:
                last_error = str(e)
            retries += 1
        return GeneratedActions(options=[], scores=[], retries=retries, raw_response=callback_handler.last_generation,
                                raw_prompt=callback_handler.last_prompt, llm_info=callback_handler.last_model_info,
                                error=f"No options found after {retries} retries."
                                                             f" Last error: {last_error}")
