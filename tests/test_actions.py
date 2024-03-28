import json

import pytest
from cattr import unstructure

from fable_saga.actions import ActionsAgent
from . import fake_actions_llm, fake_skills


class TestSagaAgent:
    def test_init(self, fake_actions_llm):
        agent = ActionsAgent(fake_actions_llm)
        assert agent._llm == fake_actions_llm

    def test_actions_chain(self, fake_actions_llm):
        agent = ActionsAgent(fake_actions_llm)
        chain = agent.generate_chain()
        assert chain.llm == fake_actions_llm
        assert chain.prompt == agent.prompt_template
        assert chain.prompt
        assert (
            "Generate a list of different action options"
            in chain.prompt.dict()["template"]
        )
        assert chain.prompt.input_variables == ["context", "skills"]

    @pytest.mark.asyncio
    async def test_generate_actions(self, fake_actions_llm, fake_skills):
        # fake_llm.callbacks = [callback_handler]
        agent = ActionsAgent(fake_actions_llm)

        # Should be using the default model
        test_model = "test_model"
        assert fake_actions_llm.model_name != test_model

        actions = await agent.generate_actions(
            "context", fake_skills, model_override=test_model
        )

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
        assert actions.raw_prompt is not None
        assert actions.raw_prompt.startswith(
            "Generate a list of different action options that your character"
            " should take next using the following skills:"
        )

        # Check that the prompt contains the right skills.
        for skill in fake_skills:
            dumped_skill = unstructure(skill)
            string_skill = json.dumps(dumped_skill)
            # assert that raw_prompt is a string
            assert actions.raw_prompt is not None and isinstance(
                actions.raw_prompt, str
            )
            assert string_skill in actions.raw_prompt

    @pytest.mark.asyncio
    async def test_generate_actions_retries(self, fake_actions_llm, fake_skills):
        fake_actions_llm.responses = ["malformed"] + fake_actions_llm.responses
        agent = ActionsAgent(fake_actions_llm)
        actions = await agent.generate_actions("context", fake_skills, max_tries=1)

        assert actions.error is None
        assert len(actions.options) == 2
        assert actions.retries == 1
