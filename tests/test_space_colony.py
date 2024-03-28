import datetime
from unittest.mock import Mock

import pytest

import fable_saga
import fable_saga.conversations
import fable_saga.actions
from fable_saga.demos.space_colony import simulation
from fable_saga.demos.space_colony.simulation import (
    ActionGenerator,
    ConversationGenerator,
)
from . import fake_actions_llm, fake_conversation_llm, FakeOpenAI


@pytest.fixture
def fake_saga_agent(fake_actions_llm):
    return fable_saga.actions.ActionsAgent(fake_actions_llm)


@pytest.fixture
def fake_conversation_agent(fake_conversation_llm):
    return fable_saga.conversations.ConversationAgent(fake_conversation_llm)


@pytest.fixture
def fake_chat_openai():
    return FakeOpenAI(responses=[], sleep=0.1)


class TestSimulation:

    def test_simulation_init(
        self, fake_saga_agent, fake_conversation_agent, fake_chat_openai
    ):
        sim = simulation.Simulation(
            ActionGenerator(fake_saga_agent),
            ConversationGenerator(fake_conversation_agent),
            fake_chat_openai,
        )
        assert sim is not None
        assert sim.sim_time == datetime.datetime(2060, 1, 1, 8, 0)

    def test_simulation_load_agents(
        self, fake_saga_agent, fake_conversation_agent, fake_chat_openai
    ):
        sim = simulation.Simulation(
            ActionGenerator(fake_saga_agent),
            ConversationGenerator(fake_conversation_agent),
            fake_chat_openai,
        )
        sim.load()

        # Check that the simulation sim_agents are initialized correctly.
        assert len(sim.agents) > 0
        captain = sim.agents["elara_sundeep"]
        assert captain.persona.first_name == "Elara"
        assert captain.persona.last_name == "Sundeep"
        assert "Born on Mars" in captain.persona.background
        assert "close-cropped haircut" in captain.persona.appearance
        assert "Diplomatic yet assertive" in captain.persona.personality
        assert "Captain" in captain.persona.role

    def test_simulation_load_skills(
        self, fake_saga_agent, fake_conversation_agent, fake_chat_openai
    ):
        sim = simulation.Simulation(
            ActionGenerator(fake_saga_agent),
            ConversationGenerator(fake_conversation_agent),
            fake_chat_openai,
        )
        sim.load()

        # Check that the skills are initialized correctly.
        captain = sim.agents["elara_sundeep"]
        assert len(captain.skills) > 0
        assert captain.skills[0].name == "go_to"
        assert list(captain.skills[0].parameters.keys()) == ["destination", "goal"]

    def test_simulation_load_locations(self, fake_saga_agent, fake_conversation_agent):
        sim = simulation.Simulation(
            ActionGenerator(fake_saga_agent),
            ConversationGenerator(fake_conversation_agent),
            fake_chat_openai,
        )
        sim.load()

        # Check that the locations are initialized correctly.
        assert len(sim.locations) > 0
        engine_room = sim.locations["engine_room"]
        assert engine_room.name == "Engine Room"
        assert "the heart of the ship" in engine_room.description
        assert engine_room.guid == "engine_room"

    def test_simulation_load_objects(
        self, fake_saga_agent, fake_conversation_agent, fake_chat_openai
    ):
        sim = simulation.Simulation(
            ActionGenerator(fake_saga_agent),
            ConversationGenerator(fake_conversation_agent),
            fake_chat_openai,
        )
        sim.load()

        # Check that the interactable objects are initialized correctly.
        assert len(sim.interactable_objects) > 0
        engine = sim.interactable_objects["captains_chair"]
        assert engine.location == "bridge"
        assert engine.affordances == ["sit", "access_ship_logs", "issue_commands"]

    @pytest.mark.asyncio
    async def test_simulation_tick(self, fake_conversation_agent, fake_chat_openai):

        goto_bridge_llm = FakeOpenAI(
            responses=[
                """{"options": [
                {"skill": "go_to", "parameters": {"destination": "bridge", "goal": "get to the bridge"}},
                {"skill": "go_to", "parameters": {"destination": "mess_hall", "goal": "eat food"}}
            ],
            "scores": [0.9, 0.1]}"""
            ]
        )

        sim = simulation.Simulation(
            ActionGenerator(fable_saga.actions.ActionsAgent(goto_bridge_llm)),
            ConversationGenerator(fake_conversation_agent),
            fake_chat_openai,
        )
        for agent in sim.agents.values():
            mock = Mock()
            # Mock the tick method, so we can check that it was called.
            agent.__setattr__("tick", mock)

        await sim.tick(datetime.timedelta(seconds=59))

        # Check that the agents were ticked.
        for agent in sim.agents.values():
            agent.tick.assert_called_once_with(datetime.timedelta(seconds=59), sim)  # type: ignore

        assert sim.sim_time == datetime.datetime(2060, 1, 1, 8, 0, 59)

    @pytest.mark.asyncio
    async def test_go_to(self, fake_conversation_llm, fake_chat_openai):

        goto_bridge_llm = FakeOpenAI(
            responses=[
                """{"options": [
                {"skill": "go_to", "parameters": {"destination": "bridge", "goal": "get to the bridge"}},
                {"skill": "go_to", "parameters": {"destination": "mess_hall", "goal": "eat food"}}
            ],
            "scores": [0.9, 0.1]}"""
            ]
        )

        sim = simulation.Simulation(
            ActionGenerator(fable_saga.actions.ActionsAgent(goto_bridge_llm)),
            ConversationGenerator(fake_conversation_llm),
            fake_chat_openai,
        )
        sim.load()

        await sim.tick(datetime.timedelta(seconds=30))
        assert sim.sim_time == datetime.datetime(2060, 1, 1, 8, 0, 30)

        # Everyone should still be in the crew quarters corridor, but have an action to go to the bridge.
        for agent in sim.agents.values():
            assert agent.action is not None
            assert agent.action.skill == "go_to"
            assert agent.location.guid == "crew_quarters_corridor"

        #  Check that everyone has moved to the bridge and don't have an action remaining as current goto always
        #  takes 1 minute.
        await sim.tick(datetime.timedelta(seconds=30))
        assert sim.sim_time == datetime.datetime(2060, 1, 1, 8, 1, 00)
        for agent in sim.agents.values():
            assert agent.action is None
            assert agent.location.guid == "bridge", (
                agent.persona.first_name + " is not in the bridge"
            )
