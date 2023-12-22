from datetime import timedelta
from typing import Dict, Any

import fable_saga
from demos.space_colony import sim_models
from demos.space_colony.simulation import SimAgent, Simulation
from demos.space_colony.sim_models import EntityId


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

    def tick(self, delta: timedelta, sim: Simulation):
        """Update the action's run time as a baseline."""
        if self.start_time is None:
            self.start_time = sim.sim_time
        else:
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

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        # Assume it takes one minute to move to the next location.
        if self.run_time.total_seconds() < 60:
            return
        previous_location = self.agent.location.guid
        self.agent.location = sim.locations[EntityId(self.destination)]
        self.agent.memories.append(sim_models.Memory(summary=f"Moved from {previous_location} to {self.destination} with goal {self.goal}",
                                                     timestamp=sim.sim_time))
        self.end_time = sim.sim_time
        print(f"{self.agent.persona.id()} moved to {self.destination}.")
        self.complete()


class Interact(SimAction):
    """Interact with an object using a specific interaction."""
    def __init__(self, agent: SimAgent, action_data: fable_saga.Action):
        super().__init__(agent, action_data)
        self.item_guid: EntityId = self.parameters.get('item_guid', None)
        self.interaction: str = self.parameters.get('interaction', None)
        self.goal: str = self.parameters.get('goal', 'None')

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        # Assume it takes one minute to interact with an object.
        if self.run_time.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Interacted with {self.item_guid} via {self.interaction} at {self.agent.location.guid} with goal {self.goal}",
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
        self.goal: str = self.parameters.get('goal', 0)

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        self.remaining_time -= delta
        # Check if we have waited long enough.
        if self.remaining_time.total_seconds() > 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Waited for {self.duration}m at {self.agent.location.guid} with goal {self.goal}",
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

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        # Assume it takes one minute to reflect on something.
        if self.run_time.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Reflected on {self.focus} and synthesized {self.result} "
                                                             f"at {self.agent.location.guid} with goal {self.goal}",
                                                     timestamp=sim.sim_time))
        self.end_time = sim.sim_time
        print(f"{self.agent.persona.id()} reflected on {self.focus} and synthesized {self.result}.")
        self.complete()


class ConverseWith(SimAction):
    """Converse with another persona."""
    def __init__(self, agent: SimAgent, action_data: fable_saga.Action):
        super().__init__(agent, action_data)

        self.persona_guid = self.parameters['persona_guid']
        self.topic = self.parameters['topic']
        self.context = self.parameters['context']
        self.goal = self.parameters.get('goal', "")  # sometimes goal doesn't get generated, so default to empty string.

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        # Assume it takes one minute to converse with someone.
        if self.run_time.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Conversed with {self.persona_guid} on {self.topic} "
                                                             f"at {self.agent.location.guid} with goal {self.goal}",
                                                     timestamp=sim.sim_time))
        sim.agents[self.persona_guid].memories.append(sim_models.Memory(summary=f"Conversed with {self.agent.persona.id()} on {self.topic} "
                                                             f"at {self.agent.location.guid}",
                                                                        timestamp=sim.sim_time))
        self.end_time = sim.sim_time
        print(f"{self.agent.persona.id()} conversed with {self.persona_guid} on {self.topic}.")
        self.complete()
