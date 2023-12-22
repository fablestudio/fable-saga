import json

import pytest
from cattr import unstructure, structure
from langchain.chat_models.fake import FakeListChatModel

import fable_saga
from fable_saga.server import SagaServer, ActionsRequest, ActionsResponse


@pytest.fixture
def fake_llm():
    actions = json.load(open("examples/generated_actions.json"))
    responses = [json.dumps(action) for action in actions]
    llm = FakeListChatModel(responses=responses, sleep=0.1)
    return llm


@pytest.fixture
def fake_skills():
    skills_data = json.load(open("examples/skills.json"))
    skills = [structure(skill_data, fable_saga.Skill) for skill_data in skills_data]
    return skills


@pytest.fixture
def fake_request():
    request_data = json.load(open("examples/request.json"))
    request = structure(request_data, ActionsRequest)
    return request


class TestAgent:
    def test_init(self, fake_llm):
        agent = fable_saga.Agent(fake_llm)
        assert agent._llm == fake_llm

    def test_chain(self, fake_llm):
        agent = fable_saga.Agent(fake_llm)
        chain = agent.chain()
        assert chain.llm == fake_llm
        assert chain.prompt == agent.prompt
        assert "Generate a list of different action options" in chain.prompt.template
        assert chain.prompt.input_variables == ["context", "skills"]

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_llm, fake_skills):

        # fake_llm.callbacks = [callback_handler]
        agent = fable_saga.Agent(fake_llm)
        actions = await agent.generate_actions("context", fake_skills)

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
    async def test_generate_actions_retries(self, fake_llm, fake_skills):
        fake_llm.responses = ["malformed"] + fake_llm.responses
        agent = fable_saga.Agent(fake_llm)
        actions = await agent.generate_actions("context", fake_skills, max_tries=1)

        assert actions.error is None
        assert len(actions.options) == 2
        assert actions.retries == 1


class TestServer:

    def test_init(self, fake_llm):
        server = SagaServer(llm=fake_llm)
        assert server.agent._llm == fake_llm

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_llm, fake_request):
        server = SagaServer(llm=fake_llm)
        response = await server.generate_actions(fake_request)

        # The response is a valid ActionsResponse
        # Note: we don't throw exceptions in the server, but return the error in the response.
        assert response.error is None

        # The reference is the same as the request.
        assert response.reference == fake_request.reference

        # Sanity check that the skills are the same in our fake_llm (test data ignores the requested skills here).
        assert isinstance(response.actions, fable_saga.GeneratedActions)
        options = response.actions.options
        assert len(options) == 2
        assert options[0].skill == "skill_2"
        assert options[1].skill == "skill_1"

    @pytest.mark.asyncio
    async def test_no_skills_error(self, fake_llm, fake_request):
        server = SagaServer(llm=fake_llm)

        # Pass an empty list of skills should raise an error.
        fake_request.skills = []
        response = await server.generate_actions(fake_request)
        assert response.error is not None
        assert response.actions is None
        assert response.reference == fake_request.reference

        assert response.error == "Must provide at least one skill."


    @pytest.mark.asyncio
    async def test_malformed_skills_error(self, fake_llm, fake_request):
        server = SagaServer(llm=fake_llm)

        # Pass an empty list of skills should raise an error.
        fake_request.skills = ["malformed"]
        response = await server.generate_actions(fake_request)
        assert response.error is not None
        assert response.actions is None
        assert response.reference == fake_request.reference

        assert response.error == "Must provide a list of Skill objects."

    @pytest.mark.asyncio
    async def test_malformed_response_error(self, fake_llm, fake_request):
        fake_llm.responses = ["malformed"]
        server = SagaServer(llm=fake_llm)

        # A malformed response should raise an error.
        response = await server.generate_actions(fake_request)
        assert response.actions is not None
        assert response.actions.error == "No options found after 1 retries. Last error: Error decoding response: Expecting value: line 1 column 1 (char 0)"
        assert response.reference == fake_request.reference
        assert response.error is not None
