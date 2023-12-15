import json
from typing import *
from attr import define
import pathlib

from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.chat_models.base import BaseChatModel
from langchain.prompts import load_prompt

from . import models
from . import datastore
from .models import EntityId

# Package wide defaults.
default_openai_model_name = "gpt-4-1106-preview"
default_openai_model_temperature = 0.9


class Agent(models.EntityInterface):

    def id(self) -> EntityId:
        return self._id

    def __init__(self, guid: EntityId, llm: BaseChatModel = None):
        self._id = guid
        self._llm = llm if llm is not None else \
            ChatOpenAI(temperature=default_openai_model_temperature, model_name=default_openai_model_name,
                       model_kwargs={
                           "response_format": {"type": "json_object"}})

    def chain(self) -> LLMChain:
        path = pathlib.Path(__file__).parent.resolve()
        prompt = load_prompt(path / "prompt_templates/generate_actions.yaml")
        return LLMChain(llm=self._llm, prompt=prompt)

    async def actions(self, context, skills, retries=0, verbose=False) -> List[Dict[str, Any]]:
        chain = self.chain()
        chain.verbose = verbose
        guidance = """
* Provide a goal for your character to achieve as well as the action.
* Do not make up new characters.
* Staying true to your character's personality and circumstances.
* Use your observations and conversations to inform your actions.
* Only use provided skills to create action options (do not make up skills).
* Do not do the same thing again if you have already done it.
* Advance the story by generating actions that will help you achieve your most important and immediate goals.
"""
        options = []
        while retries >= 0 and len(options) == 0:
            resp = await chain.arun(guidance=guidance, context=context, skills=skills)

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


@define(slots=True)
class Datastore:
    conversations: datastore.ConversationMemory = datastore.ConversationMemory()
    observation_memory: datastore.ObservationMemory = datastore.ObservationMemory()
    personas: datastore.Personas = datastore.Personas()
    meta_affordances: datastore.MetaAffordances = datastore.MetaAffordances()
    status_updates: datastore.StatusUpdates = datastore.StatusUpdates()
    sequence_updates: datastore.SequenceUpdates = datastore.SequenceUpdates()
    # memory_vectors: datastore.MemoryVectors = datastore.MemoryVectors()
    locations: datastore.Locations = datastore.Locations()
    last_player_options: Dict[str, Optional[List[Dict[str, Any]]]] = {}
    recent_goals_chosen: Dict[str, List[str]] = {}
    memories: datastore.Memories = datastore.Memories()
    extra: str = ""
