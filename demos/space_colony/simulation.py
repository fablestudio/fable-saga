import asyncio
import json
from typing import List, Dict, Optional, Coroutine, Any
import cattrs
import yaml
from datetime import datetime, timedelta
import fable_saga
from fable_saga.models import EntityId
import sim_models


class SimAgent:

    def __init__(self, guid: EntityId):
        self.guid = guid
        self.persona: Optional[sim_models.Persona] = None
        self.location: Optional[sim_models.Location] = None
        self.skills: List[sim_models.Skill] = []
        self.saga_agent: Optional[fable_saga.Agent] = None
        self.action: Optional['sim_actions.SimAction'] = None
        self.memories: List[sim_models.Memory] = []

    async def generate_actions(self, sim: 'Simulation', retries=0, verbose=False) -> [List[Dict[str, Any]]]:
        if self.saga_agent is None:
            raise ValueError("Saga agent not initialized.")

        print(f"Generating actions for {self.persona.id()} using model {self.saga_agent._llm.model_name}...")
        context = ""
        context += "CREW: The crew of the \"Stellar Runner\" is made up of the following people:\n" \
                   + f"{json.dumps([cattrs.unstructure(agent.persona) for agent in sim.agents.values()])}\n"
        context += "LOCATIONS: The following locations are available:\n" \
                   + f"{json.dumps([cattrs.unstructure(location) for location in sim.locations.values()])}\n"
        context += "MEMORIES: The following memories are available:\n" \
                   + f"{json.dumps([Format.memory(memory, sim.shiptime) for memory in self.memories])}\n"

        if self.persona is not None:
            context += f"You are a character in a story about the crew of the spaceship \"Stellar Runner\" that is travelling " \
                       + "to a space colony on one of Jupiter's moons to deliver supplies. It's still 30 days before " \
                       + "you reach your destination\n"
            context += f"You are {self.persona.id()}.\n\n"

        if self.location is not None:
            context += f"Your location is {self.location.id()}.\n"

        return await self.saga_agent.actions(context, self.skills, retries=retries, verbose=verbose)

    async def tick_action(self, delta: timedelta, sim: 'Simulation'):
        if self.action is None:
            return
        self.action.tick(delta, sim)

    async def tick(self, delta: timedelta, sim: 'Simulation'):

        def sort_actions(actions: Dict[str, Any]):
            scores = actions['scores']
            options = actions['options']
            sorted_actions = sorted(zip(scores, options), key=lambda x: x[0], reverse=True)
            actions['scores'] = [x[0] for x in sorted_actions]
            actions['options'] = [x[1] for x in sorted_actions]

        def list_actions(actions: Dict[str, Any]):
            for i, action in enumerate(actions['options']):
                output = f"#{i} -- {action['skill']} ({actions['scores'][i]})\n"
                for key, value in action['parameters'].items():
                    output += f"  {key}: {value}"
                print(output)

        def choose_action():
            item = input(f"Choose an item for {self.persona.id()}...")
            return int(item)

        def handle_action(action: Dict[str, Any]):
            from demos.space_colony.sim_actions import GoTo, Interact, ConverseWith, Wait, Reflect
            if action['skill'] == 'go_to':
                self.action = GoTo(self, action)
            elif action['skill'] == 'interact':
                self.action = Interact(self, action)
            elif action['skill'] == 'converse_with':
                self.action = ConverseWith(self, action)
            elif action['skill'] == 'wait':
                self.action = Wait(self, action)
            elif action['skill'] == 'reflect':
                self.action = Reflect(self, action)
            else:
                print(f"Unknown action {action['skill']}.")
        # Tick the current action.
        if self.action is not None:
            await self.tick_action(delta, sim)
        # Choose a new action.
        else:
            actions = await self.generate_actions(sim, verbose=False)
            print(f"\n========== {self.persona.id()} ===========")
            sort_actions(actions)
            list_actions(actions)
            while True:
                idx = choose_action()
                if 0 <= idx < len(actions['options']):
                    handle_action(actions['options'][idx])
                    print(f"Chose action {actions['options'][idx]['skill']}.")
                    break
                else:
                    print(f"Invalid choice {idx}.")


class Simulation:
    def __init__(self):
        self.shiptime = datetime(2060, 1, 1, 8, 0, 0)
        self.agents: Dict[EntityId, SimAgent] = {}
        self.locations: Dict[EntityId, sim_models.Location] = {}
        self.interactable_objects: Dict[EntityId, sim_models.InteractableObject] = {}
        self.actionsQueue: asyncio.Queue = asyncio.Queue()

    def load(self):
        with open('locations.yaml', 'r') as f:
            for location_data in yaml.load(f, Loader=yaml.FullLoader):
                location = cattrs.structure(location_data, sim_models.Location)
                self.locations[location.id()] = location

        with open('personas.yaml', 'r') as f:
            for persona_data in yaml.load(f, Loader=yaml.FullLoader):
                persona = cattrs.structure(persona_data, sim_models.Persona)
                agent = SimAgent(persona.id())
                agent.persona = persona
                # Create a saga agent for each persona.
                agent.saga_agent = fable_saga.Agent(persona.id())
                # Start everyone in the crew quarters corridor.
                agent.location = self.locations[EntityId('crew_quarters_corridor')]
                self.agents[persona.id()] = agent

        with open('interactable_objects.yaml', 'r') as f:
            for obj_data in yaml.load(f, Loader=yaml.FullLoader):
                obj = cattrs.structure(obj_data, sim_models.InteractableObject)
                self.locations[obj.id()] = obj

        with open('skills.yaml', 'r') as f:
            skills = []
            for skill_data in yaml.load(f, Loader=yaml.FullLoader):
                skills.append(cattrs.structure(skill_data, sim_models.Skill))
            for agent in self.agents.values():
                agent.skills = skills

    async def tick(self, delta: timedelta):
        self.shiptime += delta
        for agent in self.agents.values():
            self.actionsQueue.put_nowait(agent.tick(delta, self))


class Format:
    @staticmethod
    def simple_time_ago(dt: datetime, current_datetime: datetime) -> str:
        # Format as the number of days or minutes ago. Only use days if it was at least a day ago.
        days_ago = (current_datetime - dt).days
        minutes_ago = int((current_datetime - dt).total_seconds() / 60)
        if days_ago > 0:
            return str(days_ago) + ' days ago'
        else:
            return str(minutes_ago) + 'm ago'

    @staticmethod
    def memory(memory: sim_models.Memory, current_datetime: datetime) -> str:
        return f"{Format.simple_time_ago(memory.timestamp, current_datetime)}: {memory.summary}"

    @staticmethod
    def action(action: 'sim_actions.SimAction', current_datetime: datetime) -> str:
        if action is None:
            return "Idle"
        if action.start_time is None:
            timestamp = "Starting"
        else:
            timestamp = Format.simple_time_ago(action.start_time, current_datetime)
        return f"{timestamp}: {action.summary()}"


async def agent_worker(sim: Simulation):
    while True:
        agent_tick = await sim.actionsQueue.get()
        await agent_tick
        sim.actionsQueue.task_done()


# async def ainput(string):
#     from functools import partial
#     from concurrent.futures.thread import ThreadPoolExecutor
#     rie = partial(asyncio.get_event_loop().run_in_executor, ThreadPoolExecutor(1))
#     while True:
#         await rie(input)


async def main():
    sim = Simulation()
    sim.load()

    # Now, we can use the sim_storage to generate actions for each agent.
    keep_running = True
    while keep_running:
        print ("====  SHIP TIME: " + str(sim.shiptime) + "  ====")
        for agent in sim.agents.values():
            print(f"---- {agent.persona.id()} ----")
            print(f"  ROLE: {agent.persona.role}")
            print(f"  LOCATION: {agent.location.name}")
            print(f"  ACTION: {Format.action(agent.action, sim.shiptime)}")
            print("  MEMORIES:" + "".join(["\n  * " + Format.memory(m, sim.shiptime) for m in agent.memories]) + "\n")

        await sim.tick(timedelta(minutes=1))

        # Create workers to process actions.
        tasks = []
        for i in range(len(sim.agents)):
            tasks.append(asyncio.create_task(agent_worker(sim)))

        # Wait until the queue is fully processed.
        await sim.actionsQueue.join()

        # # Cancel our worker tasks.
        # for task in tasks:
        #     task.cancel()
        #
        # # Wait until all worker tasks are cancelled.
        # await asyncio.gather(*tasks, return_exceptions=True)

    print("Exiting")
    exit(0)
if __name__ == '__main__':
    asyncio.run(main())


