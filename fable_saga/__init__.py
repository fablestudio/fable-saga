import json
from abc import ABC, abstractmethod
from typing import *
import pathlib

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import load_prompt

# Package wide defaults.
default_openai_model_name = "gpt-3.5-turbo-1106"
default_openai_model_temperature = 0.9

EntityId = str


class EntityInterface(ABC):
    @abstractmethod
    def id(self) -> EntityId:
        pass


class Agent:
    """SAGA Agent """
    def id(self) -> EntityId:
        return self._id

    def __init__(self, guid: EntityId, llm: BaseChatModel = None):
        self._id = guid
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

    def chain(self) -> LLMChain:
        return LLMChain(llm=self._llm, prompt=self.prompt)

    async def actions(self, context:str, skills: List[Dict[str, Any]], retries=0, verbose=False) -> List[Dict[str, Any]]:
        chain = self.chain()
        chain.verbose = verbose

        options = []
        while retries >= 0 and len(options) == 0:
            resp = await chain.arun(guidance=self.guidance, context=context, skills=json.dumps(skills))

            try:
                options = json.loads(resp)
            except (json.JSONDecodeError, TypeError) as e:
                print("Error decoding response", e, resp)
                options = []
            if len(options) == 0:
                print("No options found. Retrying.")
                retries -= 1
            else:
                break
        return options
