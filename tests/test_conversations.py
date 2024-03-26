import json
from typing import List

import pytest
from cattrs import structure
from langchain.llms.base import BaseLanguageModel
from langchain.llms.fake import FakeListLLM

import fable_saga
import fable_saga.conversations
from fable_saga.server import ConversationRequest


# Create a fake OpenAI model that inherits from the FakeListLLM and ChatOpenAI classes.
class FakeOpenAI(FakeListLLM, BaseLanguageModel):
    model_name = "unchanged"

    def __init__(self, responses: List[str], sleep: float = 0):
        FakeListLLM.__init__(self, responses=responses, sleep=sleep)


@pytest.fixture
def fake_conversation_llm():
    conversations = json.load(open("examples/generated_conversation.json"))
    responses = [json.dumps(conversation) for conversation in conversations]
    llm = FakeOpenAI(responses=responses, sleep=0.1)
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
        assert chain.prompt == agent.prompt_template
        assert (
            "Generate a conversation by writing lines of dialogue"
            in chain.prompt.dict()["template"]
        )
        assert chain.prompt.input_variables == ["context", "persona_guids"]

    @pytest.mark.asyncio
    async def test_generate_conversation_not_openai(self, fake_conversation_llm):
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
        assert response.raw_prompt is not None
        assert response.raw_prompt.startswith("test_context\n\nGenerate a conversation")

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
