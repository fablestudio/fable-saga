import asyncio
import json
from functools import partial
from pathlib import Path
from typing import List, Callable, Optional, Awaitable, TypeVar, Type, cast, Generic
from unittest import mock

import pytest
from cattr import structure
from cattrs import structure
from langchain.embeddings.fake import DeterministicFakeEmbedding
from langchain_community.llms.fake import FakeListLLM
from langchain_core.language_models import BaseLanguageModel

import fable_saga.embeddings as saga_embed
from fable_saga.actions import Skill, ActionsAgent
from fable_saga.conversations import ConversationAgent
from fable_saga.embeddings import EmbeddingAgent
from fable_saga.server import (
    ActionsRequest,
    ConversationRequest,
    BaseEndpoint,
    ActionsResponse,
    get_generic_types,
    TReq,
    TResp,
)

path = Path(__file__).parent

# T is a type variable that is bound to BaseEndpoint or its subclasses.
TEndpoint = TypeVar("TEndpoint", bound=BaseEndpoint, covariant=True)


class FakeOpenAI(FakeListLLM, BaseLanguageModel):
    model_name = "unchanged"

    def __init__(self, responses: List[str], sleep: float = 0):
        FakeListLLM.__init__(self, responses=responses, sleep=sleep)


@pytest.fixture
def fake_actions_llm() -> FakeOpenAI:
    actions = json.load(open(path / "examples/generated_actions.json"))
    responses = [json.dumps(action) for action in actions]
    llm = FakeOpenAI(responses=responses, sleep=0.1)
    return llm


@pytest.fixture
def fake_skills() -> list[Skill]:
    skills_data = json.load(open(path / "examples/skills.json"))
    skills = [structure(skill_data, Skill) for skill_data in skills_data]
    return skills


@pytest.fixture
def fake_actions_request() -> ActionsRequest:
    request_data = json.load(open(path / "examples/actions_request.json"))
    req = structure(request_data, ActionsRequest)
    return req


@pytest.fixture
def fake_conversation_llm():
    conversations = json.load(open(path / "examples/generated_conversation.json"))
    responses = [json.dumps(conversation) for conversation in conversations]
    llm = FakeOpenAI(responses=responses, sleep=0.1)
    return llm


@pytest.fixture
def fake_conversation_request():
    request_data = json.load(open(path / "examples/conversation_request.json"))
    req = structure(request_data, ConversationRequest)
    return req


@pytest.fixture
def fake_embedding_model():
    class FakeAsyncEmbeddingModel(DeterministicFakeEmbedding):

        async def aembed_documents(self, texts: List[str]):
            func = partial(self.embed_documents, texts)
            return await asyncio.get_event_loop().run_in_executor(None, func)

        async def aembed_query(self, text: str):
            func = partial(self.embed_query, text)
            return await asyncio.get_event_loop().run_in_executor(None, func)

        def _select_relevance_score_fn(self, query: str):
            return lambda x: 0.5

    return FakeAsyncEmbeddingModel(size=1536)


@pytest.fixture
def fake_documents():
    data = json.load(open(path / "examples/embedding_documents.json"))
    documents = [saga_embed.Document(**doc) for doc in data["documents"]]
    return documents


@pytest.fixture
def fake_actions_agent(fake_actions_llm) -> ActionsAgent:
    return ActionsAgent(fake_actions_llm)


@pytest.fixture
def fake_conversation_agent(
    fake_conversation_llm,
) -> ConversationAgent:
    return ConversationAgent(fake_conversation_llm)


@pytest.fixture
def fake_embedding_agent(fake_embedding_model) -> EmbeddingAgent:
    return EmbeddingAgent(fake_embedding_model)


class EndpointMocker(Generic[TEndpoint]):

    def __init__(
        self,
        cls: Type[TEndpoint],
        override_handler: Optional[
            Callable[[TEndpoint, TReq], Awaitable[TResp]]
        ] = None,
    ):
        assert issubclass(cls, BaseEndpoint)
        self.cls = cls
        self.override_handler = override_handler
        self.original_handle_request = cls.handle_request

    def __enter__(self) -> Type[TEndpoint]:

        req_type, resp_type = get_generic_types(self.cls)

        if self.override_handler:
            setattr(
                self.cls,
                "handle_request",
                mock.create_autospec(
                    self.cls.handle_request, side_effect=self.override_handler
                ),
            )
        else:
            setattr(
                self.cls,
                "handle_request",
                mock.create_autospec(self.cls.handle_request, return_value=resp_type()),
            )
        return self.cls

    def __exit__(self, exc_type, exc_val, exc_tb):
        setattr(self.cls, "handle_request", self.original_handle_request)
