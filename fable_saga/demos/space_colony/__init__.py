from __future__ import annotations

import collections
import json
import pathlib
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Callable, Awaitable

import cattrs
import yaml
from attr import define

from langchain.llms.base import BaseLanguageModel

import fable_saga
import fable_saga.conversations
from fable_saga.actions import Action, GeneratedActions, Skill, ActionsAgent
from fable_saga.demos.space_colony import sim_models
from fable_saga.demos.space_colony.sim_actions import (
    GoTo,
    Interact,
    ConverseWith,
    Wait,
    Reflect,
    SimAction,
)

from fable_saga import logger


class MemoryStore(collections.UserList):
    """A memory store is a list of memories that can be filtered by a time range."""

    def __init__(self, memories: List[sim_models.Memory]):
        super().__init__(memories)

    def filter(self, start_time: datetime, end_time: datetime) -> "MemoryStore":
        """Filter the memories by the given time range."""
        return MemoryStore(
            [
                memory
                for memory in self.data
                if start_time <= memory.timestamp <= end_time
            ]
        )

    def find_similar(self, context, max_results=5):
        """Find the most similar memories to the given context."""
        return self.data[:max_results]


# noinspection PyShadowingNames


@define(slots=True)
class SimAgent:
    """A SimAgent is a persona in the simulation. It has a persona, a location, and a set of skills."""

    guid: str
    persona: sim_models.Persona
    location: sim_models.Location
    skills: List[Skill] = []
    action: Optional[SimAction] = None
    memories: MemoryStore = MemoryStore([])
    choose_action_callback: Optional[Callable[[GeneratedActions], Awaitable[None]]] = (
        None
    )

    async def tick_action(self, delta: timedelta, sim: "Simulation"):
        """Tick the current action to advance and/or complete it."""
        if self.action is None:
            return
        await self.action.tick(delta, sim)

    async def tick(self, delta: timedelta, sim: "Simulation"):
        """Tick the agent to advance its current action or choose a new one."""

        async def choose_action(actions: GeneratedActions):
            """Choose an action from the list by choosing a number."""
            if self.choose_action_callback is not None:
                return await self.choose_action_callback(actions)
            # otherwise, choose the top action.
            return 0

        # Choose an action if we don't have one.
        if self.action is None:
            print(
                f"\n========== {self.persona.id()} ({self.persona.job}) at {self.location.name} ==========="
            )

            actions = await sim.action_generator.generate_action_options(
                sim, self, verbose=False
            )
            if actions.error is not None:
                print(
                    f"Error generating actions: {actions.error}, waiting for next tick."
                )
                return

            actions.sort()
            while True:
                idx = await choose_action(actions)
                if 0 <= idx < len(actions.options):
                    try:
                        new_action = sim.action_generator.sim_action_factory(
                            self, actions.options[idx]
                        )
                        print(f"Chose action {new_action.skill}.")
                        self.action = new_action
                        break  # break out of the while loop, so we tick the new action immediately.
                    except Exception as e:
                        print(f"Error creating action so waiting for next tick: {e}")
                        return  # wait for the next tick.
                else:
                    print(f"Invalid choice {idx}.")

        # Tick the current action.
        await self.tick_action(delta, sim)


class Simulation:
    def __init__(
        self,
        action_generator: "ActionGenerator",
        conversation_generator: "ConversationGenerator",
        general_purpose_llm=None,
    ):
        # The current time on the ship.
        self.sim_time = datetime(2060, 1, 1, 8, 0, 0)
        # The crew on the ship.
        self.agents: Dict[sim_models.EntityId, SimAgent] = {}
        # The locations on the ship.
        self.locations: Dict[sim_models.EntityId, sim_models.Location] = {}
        # The interactable objects on the ship.
        self.interactable_objects: Dict[
            sim_models.EntityId, sim_models.InteractableObject
        ] = {}
        # The generator used to create action options.
        self.action_generator = action_generator
        # The generator used to create conversations.
        self.conversation_generator = conversation_generator
        # A general purpose langchain model for doing simple simulation hallucination.
        if general_purpose_llm is not None:
            assert isinstance(
                general_purpose_llm, BaseLanguageModel
            ), f"general_purpose_llm must be a BaseLanguageModel, but got {type(general_purpose_llm)}."
            self.sim_model = general_purpose_llm
        # If no general purpose model is provided, use the default OpenAI model.
        else:
            logger.warning("No language model provided, using default OpenAI model.")
            try:
                from langchain_openai import ChatOpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI not installed. Please install langchain-openai or provide a general purpose language model."
                )
            self.sim_model = ChatOpenAI(
                temperature=0.9,
                max_tokens=100,
                verbose=True,
            )
            if hasattr(self.sim_model, "model_name"):
                self.sim_model.model_name = "gpt4-4o"

    def load(self):
        """Load the simulation data from the YAML files."""
        path = pathlib.Path(__file__).parent.resolve()
        with open(path / "resources/locations.yaml", "r") as f:
            for location_data in yaml.load(f, Loader=yaml.FullLoader):
                location = cattrs.structure(location_data, sim_models.Location)
                self.locations[location.id()] = location

        with open(path / "resources/personas.yaml", "r") as f:
            for persona_data in yaml.load(f, Loader=yaml.FullLoader):
                persona = cattrs.structure(persona_data, sim_models.Persona)
                # Start everyone in the crew quarters corridor.
                location = self.locations[sim_models.EntityId("crew_quarters_corridor")]
                agent = SimAgent(
                    persona.id(),
                    persona=persona,
                    location=location,
                )
                self.agents[persona.id()] = agent

        with open(path / "resources/interactable_objects.yaml", "r") as f:
            for obj_data in yaml.load(f, Loader=yaml.FullLoader):
                obj = cattrs.structure(obj_data, sim_models.InteractableObject)
                self.interactable_objects[obj.id()] = obj

        with open(path / "resources/skills.yaml", "r") as f:
            skills = []
            for skill_data in yaml.load(f, Loader=yaml.FullLoader):
                skills.append(cattrs.structure(skill_data, Skill))
            for agent in self.agents.values():
                agent.skills = skills

    async def tick(self, delta: timedelta):
        """Tick the simulation to advance the current time and actions."""
        for agent in self.agents.values():
            await agent.tick(delta, self)

        # Advance the simulation time itself once everything has been ticked.
        self.sim_time += delta


class ActionGenerator:

    def __init__(self, saga_agent: ActionsAgent):
        self.saga_agent = saga_agent

    async def generate_action_options(
        self,
        sim: Simulation,
        sim_agent: SimAgent,
        retries=0,
        verbose=False,
    ) -> GeneratedActions:
        """Generate actions for this agent using the SAGA agent."""

        print(f"Generating actions for {sim_agent.persona.id()} ...")
        context = Format.standard_llm_context(sim_agent, sim)
        return await self.saga_agent.generate_actions(
            context,
            sim_agent.skills,
            max_tries=retries,
            verbose=verbose,
        )

    @staticmethod
    def sim_action_factory(sim_agent: SimAgent, action: Action):
        """Handle the chosen action."""
        if action.skill == "go_to":
            return GoTo(sim_agent, action)
        elif action.skill == "interact":
            return Interact(sim_agent, action)
        elif action.skill == "converse_with":
            return ConverseWith(sim_agent, action)
        elif action.skill == "wait":
            return Wait(sim_agent, action)
        elif action.skill == "reflect":
            return Reflect(sim_agent, action)
        else:
            raise Exception(f"Unknown action {action.skill}.")


class ConversationGenerator:

    def __init__(self, conversation_agent: fable_saga.conversations.ConversationAgent):
        self.conversation_agent = conversation_agent

    async def generate_conversation(
        self,
        sim: Simulation,
        sim_agent: SimAgent,
        other_agent_id: str,
        additional_context: Optional[str] = None,
        retries=0,
        verbose=False,
    ) -> fable_saga.conversations.GeneratedConversation:
        """Generate a conversation between sim_agent and other_agent.
        Note that saga_agent.generate_conversation can support more than 2 agents,
        but we limit this here for simplicity"""

        if sim_agent.persona is None or other_agent_id is None:
            print(
                "Error: generate_conversation requires a valid sim_agent and other_agent_id"
            )
            return fable_saga.conversations.GeneratedConversation(
                error="Invalid sim_agent or other_agent_id.", conversation=[]
            )

        context = Format.standard_llm_context(sim_agent, sim)

        context += f"[CONVERSATION]\nDo not repeat previous conversations. Check your memories for previous conversations. Write compelling dialogue!\n"
        if additional_context is not None:
            context += additional_context

        print(
            f"Generating conversation between {sim_agent.persona.id()} and {other_agent_id}..."
        )

        return await self.conversation_agent.generate_conversation(
            [sim_agent.persona.id(), other_agent_id],
            context,
            max_tries=retries,
            verbose=verbose,
        )


class Format:
    """A set of methods to format simulation data for printing to the console and to the SAGA context."""

    @staticmethod
    def simple_time_ago(dt: datetime, current_datetime: datetime) -> str:
        """Format a datetime as a simple time ago string."""
        # Format as the number of days or minutes ago. Only use days if it was at least a day ago.
        days_ago = (current_datetime - dt).days
        minutes_ago = int((current_datetime - dt).total_seconds() / 60)
        if days_ago > 0:
            return str(days_ago) + " days ago"
        else:
            return str(minutes_ago) + "m ago"

    @staticmethod
    def memory(memory: sim_models.Memory, current_datetime: datetime) -> str:
        """Format a memory with a simple time ago string."""
        return f"{Format.simple_time_ago(memory.timestamp, current_datetime)}: {memory.summary}"

    @staticmethod
    def action(action: SimAction, current_datetime: datetime) -> str:
        """Format an action with a simple time ago string."""
        if action is None:
            return "Idle"
        if action.start_time is None:
            # The action hasn't started yet.
            timestamp = "Starting"
        else:
            timestamp = Format.simple_time_ago(action.start_time, current_datetime)
        return f"{timestamp}: {action.summary()}"

    @staticmethod
    def standard_llm_context(sim_agent: SimAgent, sim: Simulation):
        context = ""
        context += (
            f"You are a character named {sim_agent.persona.id()} in a story about the crew of the spaceship"
            f' "Stellar Runner" that is travelling to a space colony on one of Jupiter\'s moons to deliver'
            f" supplies.\n"
        )
        context += (
            '[CREW]: The crew of the "Stellar Runner" is made up of the following people:\n'
            + f"{json.dumps([cattrs.unstructure(agent.persona) for agent in sim.agents.values()])}\n"
        )
        context += (
            "[LOCATIONS]: The following locations are available:\n"
            + f"{json.dumps([cattrs.unstructure(location) for location in sim.locations.values()])}\n"
        )
        context += (
            "[MEMORIES]: The following memories are available:\n"
            + f"{json.dumps([Format.memory(memory, sim.sim_time) for memory in sim_agent.memories])}\n"
        )
        context += (
            "[INTERACTABLE OBJECTS]: The following interactable objects are available:\n"
            + f"{json.dumps([cattrs.unstructure(obj) for obj in sim.interactable_objects.values()])}\n"
        )
        context += (
            "[SKILLS]: The following skills are available to generate actions from:\n"
            + f"{json.dumps([cattrs.unstructure(skill) for skill in sim_agent.skills])}\n"
        )

        context += f"[CONTEXT]" + (
            f"You are a character performing in a larger narrative, not a real person. Choose actions that will "
            f"help you achieve your most important and immediate goals. Do not do the same thing again if you "
            f"have already done it. Staying true to your character's personality and circumstances. "
            f"Use your observations and conversations to inform your actions. Do not make up new characters. "
            f"Only use provided skills to create action options (do not make up skills). "
            f"Provide a goal for your character to achieve as well as the action. "
            f"Keep things interesting! Move the narrative forward! Add Drama! Add Conflict! Add Tension! Add "
            f"Suspense! Add Mystery! Add Intrigue! Add Romance! Add Comedy! Add Action! Add Adventure! Add Horror! "
            f"Write yor actions as if you were writing a story. Give them depth and backstory. Make them disagreeable"
        )

        if sim_agent.location is not None:
            context += f"Your location is {sim_agent.location.id()}.\n"
        return context


async def main():
    """The main entry point for the simulation."""
    sim: Simulation = Simulation(
        ActionGenerator(ActionsAgent()),
        ConversationGenerator(fable_saga.conversations.ConversationAgent()),
    )
    sim.load()

    async def list_actions(actions: GeneratedActions):
        """List the actions byt printing to the console."""
        for i, action in enumerate(actions.options):
            output = f"#{i} -- {action.skill} ({actions.scores[i]})\n"
            for key, value in action.parameters.items():
                output += f"  {key}: {value}"
            print(output)
        value = input("Choose an action: ")
        if value == "" or value is None:
            return 0
        while True:
            try:
                assert 0 <= int(value) < len(actions.options)
                return int(value)
            except (ValueError, AssertionError) as e:
                print(f"Invalid choice {value}: {e}.")
                value = input("Choose an action: ")

    for agent in sim.agents.values():
        agent.choose_action_callback = list_actions

    keep_running = True
    while keep_running:
        print("====  SHIP TIME: " + str(sim.sim_time) + "  ====")
        for agent in sim.agents.values():
            print(f"---- {agent.persona.id()} ----")
            print(f"  ROLE: {agent.persona.role}")
            print(f"  LOCATION: {agent.location.name}")
            if agent.action is not None:
                print(f"  ACTION: {Format.action(agent.action, sim.sim_time)}")
            print(
                "  MEMORIES:"
                + "".join(
                    ["\n  * " + Format.memory(m, sim.sim_time) for m in agent.memories]
                )
                + "\n"
            )

        await sim.tick(timedelta(minutes=1))

    print("Exiting")
    exit(0)
