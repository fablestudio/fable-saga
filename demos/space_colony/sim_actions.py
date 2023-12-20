from datetime import timedelta
from typing import Dict, Any

from demos.space_colony import sim_models
from demos.space_colony.simulation import SimAgent, Simulation
from fable_saga import EntityId


class SimAction:
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
        self.agent = agent
        self.action_data: Dict[str, Any] = action_data
        self.skill: str = action_data['skill']
        self.parameters: Dict[str, Any] = action_data.get('parameters', {})
        self.start_time = None
        self.run_time = timedelta()
        self.end_time = None

    def tick(self, delta: timedelta, sim: Simulation):
        if self.start_time is None:
            self.start_time = sim.shiptime
        else:
            self.run_time += delta

    def complete(self):
        self.agent.action = None

    def summary(self):
        output = self.skill
        for key, value in self.parameters.items():
            output += f", {key}:{value}"
        return output


class GoTo(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
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
                                                    timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} moved to {self.destination}.")
        self.complete()


class Interact(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
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
                                                    timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} interacted with {self.item_guid}.")
        self.complete()


class Wait(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
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
                                                    timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} waited for {self.duration}.")
        self.complete()


class Reflect(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
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
                                                    timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} reflected on {self.focus} and synthesized {self.result}.")
        self.complete()


class ConverseWith(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
        super().__init__(agent, action_data)

        self.persona_guid = action_data['parameters']['persona_guid']
        self.topic = action_data['parameters']['topic']
        self.context = action_data['parameters']['context']
        self.goal = action_data['parameters']['goal']

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        # Assume it takes one minute to converse with someone.
        if self.run_time.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Conversed with {self.persona_guid} on {self.topic} "
                                                             f"at {self.agent.location.guid} with goal {self.goal}",
                                                    timestamp=sim.shiptime))
        sim.agents[self.persona_guid].memories.append(sim_models.Memory(summary=f"Conversed with {self.agent.persona.id()} on {self.topic} "
                                                             f"at {self.agent.location.guid}",
                                                     timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} conversed with {self.persona_guid} on {self.topic}.")
        self.complete()
