from __future__ import annotations

from datetime import timedelta
import typing
from typing import Dict, Any

import langchain

import fable_saga
from demos.space_colony import sim_models
from demos.space_colony.sim_models import EntityId

if typing.TYPE_CHECKING:
    from demos.space_colony.simulation import Simulation, SimAgent


class SimAction:
    """Base class for all actions."""

    def __init__(self, agent: SimAgent, action_data: fable_saga.Action):
        self.agent = agent
        self.action_data: fable_saga.Action = action_data
        self.skill: str = action_data.skill
        self.parameters: Dict[str, Any] = action_data.parameters
        self.start_time = None
        self.run_time = timedelta()
        self.end_time = None

    async def tick(self, delta: timedelta, sim: Simulation):
        """Update the action's run time as a baseline."""
        if self.start_time is None:
            self.start_time = sim.sim_time
        self.run_time += delta

    def complete(self):
        """Mark the action as complete by resetting the agent's action to None."""
        self.agent.action = None

    def summary(self):
        """Return a summary of the action."""
        output = self.skill
        for key, value in self.parameters.items():
            output += f", {key}:{value}"
        return output


class GoTo(SimAction):
    """Move to a new location."""

    def __init__(self, agent: SimAgent, action_data: fable_saga.Action):
        super().__init__(agent, action_data)
        self.destination: EntityId = self.parameters.get('destination', "")
        self.goal: str = self.parameters.get('goal', "None")

    async def tick(self, delta: timedelta, sim: Simulation):
        await super().tick(delta, sim)
        # Assume it takes one minute to move to the next location.
        if self.run_time.total_seconds() < 60:
            return
        previous_location = self.agent.location.guid
        self.agent.location = sim.locations[EntityId(self.destination)]

        summary = f"Moved from {previous_location} to {self.destination} with goal {self.goal}"

        self.agent.memories.append(sim_models.Memory(summary=summary,
                                                     timestamp=sim.sim_time))
        self.end_time = sim.sim_time
        print(f"{self.agent.persona.id()}: {summary}")
        self.complete()


class Interact(SimAction):
    """Interact with an object using a specific interaction."""

    def __init__(self, agent: SimAgent, action_data: fable_saga.Action):
        super().__init__(agent, action_data)
        self.item_guid: EntityId = self.parameters.get('item_guid')
        self.interaction: str = self.parameters.get('interaction')
        self.goal: str = self.parameters.get('goal', 'None')

    async def tick(self, delta: timedelta, sim: Simulation):
        await super().tick(delta, sim)
        # Assume it takes one minute to interact with an object.
        if self.run_time.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Interacted with {self.item_guid} via {self.interaction}"
                                                             f" at {self.agent.location.guid} with goal {self.goal}",
                                                     timestamp=sim.sim_time))
        self.end_time = sim.sim_time
        print(f"{self.agent.persona.id()} interacted with {self.item_guid}.")
        self.complete()


class Wait(SimAction):
    """Wait for a period of time."""

    def __init__(self, agent: SimAgent, action_data: fable_saga.Action):
        super().__init__(agent, action_data)
        self.duration: int = int(self.parameters.get('duration', 0))
        self.remaining_time: timedelta = timedelta(minutes=self.duration)
        self.goal: str = self.parameters.get('goal', "")

    async def tick(self, delta: timedelta, sim: Simulation):
        await super().tick(delta, sim)
        self.remaining_time -= delta
        # Check if we have waited long enough.
        if self.remaining_time.total_seconds() >= 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Waited for {self.duration}m "
                                                             f"at {self.agent.location.guid} "
                                                             f"with goal {self.goal}",
                                                     timestamp=sim.sim_time))
        self.end_time = sim.sim_time
        print(f"{self.agent.persona.id()} waited for {self.duration}.")
        self.complete()


class Reflect(SimAction):
    """Reflect on something and synthesize a new idea."""

    def __init__(self, agent: SimAgent, action_data: fable_saga.Action):
        super().__init__(agent, action_data)
        self.focus: str = self.parameters.get('focus', "")
        self.result: str = self.parameters.get('result', "")
        self.goal: str = self.parameters.get('goal', "")

    async def tick(self, delta: timedelta, sim: Simulation):
        await super().tick(delta, sim)
        # Assume it takes one minute to reflect on something.
        if self.run_time.total_seconds() < 60:
            return
            # Generate and format the conversation into a memory
        import simulation
        response = sim.sim_model.invoke(
            simulation.Format.standard_llm_context(self.agent, sim) +
            f"In the context of {self.focus}, generate a VERY short, one sentence, specific actionable plan to achieve {self.result} within a larger goal of {self.goal}."
            f"Then, break down the plan into smaller steps that ONLY use the types of skills listed as numbered bullet points and are also very, very short. Don't explain the steps, just list them"
            f"For example, if the focus is 'ship', the result is 'fix engine', and the goal is 'escape attack', then the plan might be 'replace coupler to fix ship engine'."
            f"Then, the steps might be '1) take_to: coupler - to engine_room', '2) interact: engine - replace engine coupler', 3) converse_with: captain - confirm it's fixed."
        )
        reflection = response.content

        formatted_reflection = f"Reflected while " + \
                               f"at {self.agent.location.guid}: " + \
                               f"{reflection}"

        self.agent.memories.append(sim_models.Memory(summary=formatted_reflection,
                                                     timestamp=sim.sim_time))
        self.end_time = sim.sim_time
        print(f"{self.agent.persona.id()}: {formatted_reflection}")
        self.complete()


class ConverseWith(SimAction):
    """Converse with another persona."""

    def __init__(self, agent: SimAgent, action_data: fable_saga.Action):
        super().__init__(agent, action_data)

        self.persona_guid = self.parameters['persona_guid']  # The other person to speak with
        self.topic = self.parameters['topic']
        self.context = self.parameters['context']
        self.goal = self.parameters.get('goal', "")  # sometimes goal doesn't get generated, so default to empty string.

    async def tick(self, delta: timedelta, sim: Simulation):
        await super().tick(delta, sim)

        # Generate and format the conversation into a memory
        generated_conversation = await sim.conversation_generator.generate_conversation(sim, self.agent,
                                                                                        self.persona_guid,
                                                                                        f"[TOPIC] \n {self.topic}")
        if generated_conversation.error is None:
            conversation_formatted = "Dialogue:\n"
            for turn in generated_conversation.conversation:
                conversation_formatted += f"\t{turn.persona_guid}: {turn.dialogue}\n"

            self.agent.memories.append(sim_models.Memory(
                summary=f"Conversed with {self.persona_guid} on {self.topic} "
                        f"at {self.agent.location.guid} with goal {self.goal}. "
                        f"{conversation_formatted}", timestamp=sim.sim_time))
            sim.agents[self.persona_guid].memories.append(sim_models.Memory(
                summary=f"Conversed with {self.agent.persona.id()} "
                        f"on {self.topic}. {conversation_formatted}"
                        f"at {self.agent.location.guid}", timestamp=sim.sim_time))
            self.end_time = sim.sim_time
            print(f"{self.agent.persona.id()} conversed with {self.persona_guid} on {self.topic}.\n"
                  f"{conversation_formatted}")
        else:
            print(f"Failed to generate conversation: {generated_conversation.error}")

        self.complete()
