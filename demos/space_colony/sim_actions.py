from datetime import timedelta
from typing import Dict, Any

from demos.space_colony import sim_models
from demos.space_colony.simulation import SimAgent, Simulation
from fable_saga import EntityId


class SimAction:
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
        self.agent = agent
        self.action = action_data
        self.start_time = None
        self.duration = timedelta()
        self.end_time = None

    def tick(self, delta: timedelta, sim: Simulation):
        if self.start_time is None:
            self.start_time = sim.shiptime
        else:
            self.duration += delta

    def complete(self):
        self.agent.action = None


class GoTo(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
        super().__init__(agent, action_data)
        self.destination = action_data['parameters']['destination']
        self.goal = action_data['parameters']['goal']

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        # Assume it takes one minute to move to the next location.
        if self.duration.total_seconds() < 60:
            return
        previous_location = self.agent.location
        self.agent.location = sim.locations[EntityId(self.destination)]
        self.agent.memories.append(sim_models.Memory(summary=f"Moved to {self.destination} from {previous_location} with goal {self.goal}",
                                                    timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} moved to {self.destination}.")
        self.complete()


class Interact(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
        super().__init__(agent, action_data)
        self.item_guid = action_data['parameters']['item_guid']
        self.interaction = action_data['parameters']['interaction']
        self.goal = action_data['parameters']['goal']

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        # Assume it takes one minute to interact with an object.
        if self.duration.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Interacted with {self.item_guid} via {self.interaction} at {self.agent.location} with goal {self.goal}",
                                                    timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} interacted with {self.item_guid}.")
        self.complete()


class Wait(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
        super().__init__(agent, action_data)
        self.duration = timedelta(minutes=int(action_data['parameters']['duration']))
        self.goal = action_data['parameters']['goal']

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        if self.duration.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Waited for {self.duration}m at {self.agent.location} with goal {self.goal}",
                                                    timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} waited for {self.duration}.")
        self.complete()


class Reflect(SimAction):
    def __init__(self, agent: SimAgent, action_data: Dict[str, Any]):
        super().__init__(agent, action_data)
        self.focus = action_data['parameters']['focus']
        self.result = action_data['parameters']['result']
        self.goal = action_data['parameters']['goal']

    def tick(self, delta: timedelta, sim: Simulation):
        super().tick(delta, sim)
        if self.duration.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Reflected on {self.focus} and synthesized {self.result} "
                                                             f"at {self.agent.location} with goal {self.goal}",
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
        if self.duration.total_seconds() < 60:
            return
        self.agent.memories.append(sim_models.Memory(summary=f"Conversed with {self.persona_guid} on {self.topic} "
                                                             f"at {self.agent.location} with goal {self.goal}",
                                                    timestamp=sim.shiptime))
        sim.agents[self.persona_guid].memories.append(sim_models.Memory(summary=f"Conversed with {self.agent.persona.id()} on {self.topic} "
                                                             f"at {self.agent.location}",
                                                     timestamp=sim.shiptime))
        self.end_time = sim.shiptime
        print(f"{self.agent.persona.id()} conversed with {self.persona_guid} on {self.topic}.")
        self.complete()
