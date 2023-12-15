import asyncio
import json
from typing import List, Dict, Optional, Coroutine, Any
from attrs import define
import cattrs
import yaml
from datetime import datetime
import fable_saga
from fable_saga.models import EntityId
import sim_models


@define(slots=True)
class SimAgent:
    persona: sim_models.Persona
    location: Optional[sim_models.Location] = None
    skills: List[sim_models.Skill] = []
    saga_agent: Optional[fable_saga.Agent] = None

    async def generate_actions(self, context, retries=0, verbose=False) -> [List[Dict[str, Any]]]:
        if self.saga_agent is None:
            raise ValueError("Saga agent not initialized.")

        if self.persona is not None:
            context += f"You are a character in a story about the crew of the spaceship \"Stellar Runner\" that is travelling " \
                       + "to a space colony on one of Jupiter's moons to deliver supplies. It's still 30 days before " \
                       + "you reach your destination\n"
            context += f"You are {self.persona.id()}.\n\n"

        if self.location is not None:
            context += f"Your location is {self.location.id()}.\n"

        return await self.saga_agent.actions(context, self.skills, retries=retries, verbose=verbose)


@define(slots=True)
class SimStorage:
    datetime: datetime = datetime.now()
    agents: Dict[EntityId, SimAgent] = {}
    locations: Dict[EntityId, sim_models.Location] = {}
    interactable_objects: Dict[EntityId, sim_models.InteractableObject] = {}


def load(storage: SimStorage):
    with open('locations.yaml', 'r') as f:
        for location_data in yaml.load(f, Loader=yaml.FullLoader):
            location = cattrs.structure(location_data, sim_models.Location)
            storage.locations[location.id()] = location

    with open('personas.yaml', 'r') as f:
        for persona_data in yaml.load(f, Loader=yaml.FullLoader):
            persona = cattrs.structure(persona_data, sim_models.Persona)
            agent = SimAgent(persona.id())
            agent.persona = persona
            # Create a saga agent for each persona.
            agent.saga_agent = fable_saga.Agent(persona.id())
            # Start everyone in the crew quarters corridor.
            agent.location = storage.locations[EntityId('crew_quarters_corridor')]
            storage.agents[persona.id()] = agent

    with open('interactable_objects.yaml', 'r') as f:
        for obj_data in yaml.load(f, Loader=yaml.FullLoader):
            obj = cattrs.structure(obj_data, sim_models.InteractableObject)
            storage.locations[obj.id()] = obj

    with open('skills.yaml', 'r') as f:
        skills = []
        for skill_data in yaml.load(f, Loader=yaml.FullLoader):
            skills.append(cattrs.structure(skill_data, sim_models.Skill))
        for agent in storage.agents.values():
            agent.skills = skills


async def main():
    sim_storage = SimStorage()
    load(sim_storage)

    # Now, we can use the sim_storage to generate actions for each agent.
    keep_running = True
    while keep_running:

        context = ""
        context += "CREW: The crew of the LeCun is made up of the following people:\n" \
                   + f"{json.dumps([cattrs.unstructure(agent.persona) for agent in sim_storage.agents.values()])}\n"
        context += "LOCATIONS: The following locations are available:\n" \
                   + f"{json.dumps([cattrs.unstructure(location) for location in sim_storage.locations.values()])}\n"

        for agent in sim_storage.agents.values():
            print(f"Generating actions for {agent.persona.id()} using model {agent.saga_agent._llm.model_name}...")
            actions = await agent.generate_actions(context, verbose=True)
            for i, action in enumerate(actions['options']):
                print(f"#{i} --{actions['scores'][i]}--\n{json.dumps(action, indent=4)}\n------")

            item = input("Choose an item or press (q) to quit...")
            if item == 'q':
                keep_running = False
                break
            else:
                item = int(item)
                if item >= 0 and item < len(actions):
                    action = actions[item]
                    print(f"Executing action {action['action']}...")
                    if action['action'] == 'go_to':
                        agent.location = sim_storage.locations[EntityId(action['parameters']['destination'])]
                    elif action['action'] == 'interact':
                        agent.location = sim_storage.locations[EntityId(action['parameters']['item_guid'])]
                    else:
                        print(f"Unknown action {action['action']}.")
                else:
                    print(f"Invalid action {item}.")

        keep_running = False

if __name__ == '__main__':
    asyncio.run(main())


