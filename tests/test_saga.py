import json

import pytest
from cattr import unstructure, structure
from langchain.chat_models.fake import FakeListChatModel

import fable_saga
from fable_saga.server import ActionsRequest, ConversationRequest


class FakeChatOpenAI(FakeListChatModel):
    model_name: str = 'model_name_default'


@pytest.fixture
def fake_actions_llm():
    actions = json.load(open("examples/generated_actions.json"))
    responses = [json.dumps(action) for action in actions]
    llm = FakeChatOpenAI(responses=responses, sleep=0.1)
    return llm


@pytest.fixture
def fake_conversation_llm():
    conversations = json.load(open("examples/generated_conversation.json"))
    responses = [json.dumps(conversation) for conversation in conversations]
    llm = FakeChatOpenAI(responses=responses, sleep=0.1)
    return llm


@pytest.fixture
def fake_skills():
    skills_data = json.load(open("examples/skills.json"))
    skills = [structure(skill_data, fable_saga.Skill) for skill_data in skills_data]
    return skills


@pytest.fixture
def fake_actions_request():
    request_data = json.load(open("examples/actions_request.json"))
    req = structure(request_data, ActionsRequest)
    return req


@pytest.fixture
def fake_conversation_request():
    request_data = json.load(open("examples/conversation_request.json"))
    req = structure(request_data, ConversationRequest)
    return req


class TestSagaAgent:
    def test_init(self, fake_actions_llm):
        agent = fable_saga.Agent(fake_actions_llm)
        assert agent._llm == fake_actions_llm

    def test_actions_chain(self, fake_actions_llm):
        agent = fable_saga.Agent(fake_actions_llm)
        chain = agent.generate_actions_chain()
        assert chain.llm == fake_actions_llm
        assert chain.prompt == agent.generate_actions_prompt
        assert "Generate a list of different action options" in chain.prompt.template
        assert chain.prompt.input_variables == ["context", "skills"]

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_actions_llm, fake_skills):

        # fake_llm.callbacks = [callback_handler]
        agent = fable_saga.Agent(fake_actions_llm)

        # Should be using the default model
        test_model = 'test_model'
        assert fake_actions_llm.model_name != test_model

        actions = await agent.generate_actions("context", fake_skills, model_override=test_model)

        # Should be using the test model
        assert fake_actions_llm.model_name == test_model

        # In our test data, we assume 2 actions are generated and are pre-sorted by score.
        assert len(actions.options) == 2
        assert len(actions.scores) == 2
        assert actions.options[0].skill == "skill_2"
        assert actions.scores[0] == 0.9
        assert actions.options[1].skill == "skill_1"
        assert actions.scores[1] == 0.1

        # Check that the prompt starts with the right text.
        # Note: We don't add the "Human: " prefix in the test data, LangChain does that.
        assert actions.raw_prompt.startswith("Human: Generate a list of different action options that your character"
                                             " should take next using the following skills:")

        # Check that the prompt contains the right skills.
        for skill in fake_skills:
            dumped_skill = unstructure(skill)
            string_skill = json.dumps(dumped_skill)
            # assert that raw_prompt is a string
            assert actions.raw_prompt is not None and isinstance(actions.raw_prompt, str)
            assert string_skill in actions.raw_prompt

    @pytest.mark.asyncio
    async def test_generate_actions_retries(self, fake_actions_llm, fake_skills):
        fake_actions_llm.responses = ["malformed"] + fake_actions_llm.responses
        agent = fable_saga.Agent(fake_actions_llm)
        actions = await agent.generate_actions("context", fake_skills, max_tries=1)

        assert actions.error is None
        assert len(actions.options) == 2
        assert actions.retries == 1

    def test_conversation_chain(self, fake_conversation_llm):
        agent = fable_saga.Agent(fake_conversation_llm)
        chain = agent.generate_conversation_chain()
        assert chain.llm == fake_conversation_llm
        assert chain.prompt == agent.generate_conversation_prompt
        assert "Generate a conversation between characters with the following identifiers" in chain.prompt.template
        assert chain.prompt.input_variables == ["context", "personas"]

    @pytest.mark.asyncio
    async def test_generate_conversation(self, fake_conversation_llm):

        # fake_llm.callbacks = [callback_handler]
        agent = fable_saga.Agent(fake_conversation_llm)

        # Should be using the default model
        test_model = 'test_model'
        assert fake_conversation_llm.model_name != test_model

        response = await agent.generate_conversation(["person_a", "person_b"], "context", model_override=test_model)
        assert isinstance(response, fable_saga.GeneratedConversation)

        # Should be using the test model
        assert fake_conversation_llm.model_name == test_model

        # Validate conversation output
        assert len(response.conversation) == 2
        assert response.conversation[0].persona_id == "person_a"
        assert response.conversation[0].dialogue == "person_a_dialogue"
        assert response.conversation[1].persona_id == "person_b"
        assert response.conversation[1].dialogue == "person_b_dialogue"

        # Check that the prompt starts with the right text.
        # Note: We don't add the "Human: " prefix in the test data, LangChain does that.
        assert response.raw_prompt.startswith("Human: Generate a conversation between characters with the following identifiers")

    @pytest.mark.asyncio
    async def test_generate_conversation_retries(self, fake_conversation_llm):
        fake_conversation_llm.responses = ["malformed"] + fake_conversation_llm.responses
        agent = fable_saga.Agent(fake_conversation_llm)
        response = await agent.generate_conversation(["person_a", "person_b"], "context", max_tries=1)

        assert response.error is None
        assert len(response.conversation) == 2
        assert response.retries == 1
