import json
from typing import Dict, Any, Union, List

import pytest
from langchain.callbacks.base import BaseCallbackHandler
from langchain.chat_models.fake import FakeListChatModel
from langchain.schema import AgentAction

import fable_saga


@pytest.fixture
def fake_llm():
    actions = json.load(open("examples/generated_actions.json"))
    responses = [json.dumps(action) for action in actions]
    llm = FakeListChatModel(responses=responses, sleep=0.1, callbacks=[TestCallbackHandler()])
    return llm

@pytest.fixture
def fake_skills():
    skills = json.load(open("examples/skills.json"))
    return skills


class TestCallbackHandler(BaseCallbackHandler):
    """Async callback handler that can be used to handle callbacks from langchain."""

    def __init__(self):
        super().__init__()
        self.last_prompts: Dict[str, Any] = {}
        self.calls: int = 0

    def on_llm_start(
            self, serialized: Dict[str, Any], prompts: List[str], **kwargs: Any
    ) -> Any:
        self.last_prompts = prompts
        self.calls += 1


class TestAgent:
    def test_init(self, fake_llm):
        agent = fable_saga.Agent("guid", fake_llm)
        assert agent.id() == "guid"

    def test_chain(self, fake_llm):
        agent = fable_saga.Agent("guid", fake_llm)
        chain = agent.chain()
        assert chain.llm == fake_llm
        assert chain.prompt == agent.prompt
        assert "Generate a list of different action options" in chain.prompt.template
        assert chain.prompt.input_variables == ["context", "skills"]

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_llm, fake_skills):

        # fake_llm.callbacks = [callback_handler]
        agent = fable_saga.Agent("guid", fake_llm)
        actions = await agent.actions("context", fake_skills)
        assert len(actions) == 2
        assert agent._llm.callbacks[0].calls == 1
        assert agent._llm.callbacks[0].last_prompts[0].startswith("Human: Generate a list of different action options that your character should take next using the following skills:")
        assert json.dumps(fake_skills) in agent._llm.callbacks[0].last_prompts[0]