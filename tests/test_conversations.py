import json

import pytest
from cattrs import structure
from langchain.chat_models.fake import FakeListChatModel

import fable_saga
import fable_saga.conversations
from fable_saga.server import ConversationRequest


class FakeChatOpenAI(FakeListChatModel):
    model_name: str = "model_name_default"


@pytest.fixture
def fake_conversation_llm():
    conversations = json.load(open("examples/generated_conversation.json"))
    responses = [json.dumps(conversation) for conversation in conversations]
    llm = FakeChatOpenAI(responses=responses, sleep=0.1)
    return llm


@pytest.fixture
def fake_conversation_request():
    request_data = json.load(open("examples/conversation_request.json"))
    req = structure(request_data, ConversationRequest)
    return req


class TestConversationAgent:
    def test_conversation_chain(self, fake_conversation_llm):
        agent = fable_saga.conversations.ConversationAgent(fake_conversation_llm)
        chain = agent.generate_chain()
        assert chain.llm == fake_conversation_llm
        assert chain.prompt == agent.generate_conversation_prompt
        assert (
            "Generate a conversation by writing lines of dialogue"
            in chain.prompt.template
        )
        assert chain.prompt.input_variables == ["context", "persona_guids"]

    @pytest.mark.asyncio
    async def test_generate_conversation(self, fake_conversation_llm):
        # fake_llm.callbacks = [callback_handler]
        agent = fable_saga.conversations.ConversationAgent(fake_conversation_llm)

        # Should be using the default model
        test_model = "test_model"
        assert fake_conversation_llm.model_name != test_model

        response = await agent.generate_conversation(
            ["person_a", "person_b"], "test_context", model_override=test_model
        )
        assert isinstance(response, fable_saga.conversations.GeneratedConversation)

        # Should be using the test model
        assert fake_conversation_llm.model_name == test_model

        # Validate conversation output
        assert len(response.conversation) == 2
        assert response.conversation[0].persona_guid == "person_a"
        assert response.conversation[0].dialogue == "person_a_dialogue"
        assert response.conversation[1].persona_guid == "person_b"
        assert response.conversation[1].dialogue == "person_b_dialogue"

        # Check that the prompt starts with the right text.
        # Note: We don't add the "Human: " prefix in the test data, LangChain does that.
        assert response.raw_prompt.startswith("Human: test_context")

    @pytest.mark.asyncio
    async def test_generate_conversation_retries(self, fake_conversation_llm):
        fake_conversation_llm.responses = [
            "malformed"
        ] + fake_conversation_llm.responses
        agent = fable_saga.conversations.ConversationAgent(fake_conversation_llm)
        response = await agent.generate_conversation(
            ["person_a", "person_b"], "context", max_tries=1
        )

        assert response.error is None
        assert len(response.conversation) == 2
        assert response.retries == 1
