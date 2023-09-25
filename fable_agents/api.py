import json
from dataclasses import dataclass
import random
from pprint import pprint
from typing import List, Callable, Optional, Any
from models import Persona, Vector3, StatusUpdate, Message
from datastore import memory_datastore
import socketio
import asyncio
import time

from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.prompts import load_prompt


sio: socketio.AsyncServer = None
simulation_client_id: Optional[str] = None

debug=True


class Format:
    @staticmethod
    def persona(persona: Persona):
        return {
            'ID': persona.guid,
            'FirstName': persona.first_name,
            'LastName': persona.last_name,
            'Description': persona.description,
            'Summary': persona.summary,
            'BackStory': persona.backstory
        }

    @staticmethod
    def observation_event(event: StatusUpdate, observer_location: Vector3):
        distance = Vector3.distance(event.location, observer_location)
        if (distance):
            distance = str(round(distance, 1)) + "m"
        return {
            'persona_guid': event.guid,
            'action': event.sequence,
            'action_step': event.sequence_step,
            'distance': distance,
        }

class GaiaAPI:

    observation_distance = 10
    observation_limit = 10

    async def create_conversation(self, persona_guid:str, on_complete: Callable[[Message], None]):
        """
        Create a new conversation.
        :param persona_guids: unique identifiers for the personas.
        :param callback: function to call when the conversation is created.
        """
        initiator_persona = memory_datastore.personas.get(persona_guid, None)
        if initiator_persona is None:
            print('Error: persona not found.')
            if on_complete is not None:
                on_complete(Message('error', {'error': 'persona not found.'}))
            return

        # Create a list of personas to send to the server as options.
        options = []

        # TODO: Using all personas results in too much text.
        # pick 5 random personas
        items = list(memory_datastore.personas.items())
        random.shuffle(items)
        for guid, persona_option in items[:5]:
            options.append(Format.persona(persona_option))

        prompt = load_prompt("prompt_templates/conversation_v1.yaml")
        llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")
        chain = LLMChain(llm=llm, prompt=prompt)
        resp = await chain.arun(self_description=json.dumps(Format.persona(initiator_persona)), options=json.dumps(options))
        if on_complete is not None:
            on_complete(resp)

    async def create_observations(self, observer_update: StatusUpdate, updates_to_consider: List[StatusUpdate], on_complete: Callable[[Message], None]):
        """
        Create a new observation.
        :param persona_guids: unique identifiers for the personas.
        :param callback: function to call when the conversation is created.
        """
        initiator_persona = memory_datastore.personas.get(observer_update.guid, None)
        if initiator_persona is None:
            print('Error: persona not found.')
            if on_complete is not None:
                on_complete(Message('error', {'error': 'persona not found.'}))
            return

        # Create a list of personas to send to the server as options.
        observation_events = []

        # sort by distance for now. Later we might include a priority metric as well.
        updates_to_consider.sort(key=lambda x: Vector3.distance(x.location, observer_update.location))
        for update in updates_to_consider[:self.observation_limit]:
            if update.location is not None and Vector3.distance(update.location, observer_update.location) <= self.observation_distance:
                observation_events.append(Format.observation_event(update, observer_update.location))

        prompt = load_prompt("prompt_templates/observation_v1.yaml")
        llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")
        chain = LLMChain(llm=llm, prompt=prompt)
        pprint(observation_events)
        resp = await chain.arun(self_description=json.dumps(Format.persona(initiator_persona)),
                                self_update=json.dumps(Format.observation_event(observer_update, observer_update.location)),
                                update_options=json.dumps(observation_events))
        if on_complete is not None:
            on_complete(resp)


class SimulationAPI:

    async def reload_personas(self, guids: List[str], on_complete: Optional[Callable[[], None]]):
        """
        Load persona's current state (i.e. description and memory).
        :param on_complete:
        :param guids: unique identifiers for the persona.
        :param callback: function to call when the personas are loaded.
        """

        def convert_to_personas(response: Message):
            print("RESPONSE", response)
            if (response.type != 'request-personas-response'):
                print('Error: expected personas response.', response)
                if on_complete is not None:
                    on_complete()
                return
            for json_rep in response.data['personas']:
                persona = Persona.from_json(json_rep)
                memory_datastore.personas[persona.guid] = persona

            # TODO: This should return when this process is complete.
            if on_complete is not None:
                on_complete()

        # Note: Server callback only works with a specific client id.
        await self.send('request-personas', {'guids': guids}, callback=convert_to_personas)

    async def send(self, type: str, data: dict, callback: Callable[[Message], None]):
        """
        Send a request to the server.
        :param type: type of request.
        :param data: data to send with the request.
        :param callback: function to call when the server responds.
        """

        def convert_response_to_message(response_type: str, response_data: str):
            response_message = Message(response_type, json.loads(response_data))
            callback(response_message)

        if simulation_client_id is None:
            print('Error: simulation_client_id is None.')
            if callback is not None:
                callback(Message('error', {'error': 'simulation_client_id is None.'}))
            return
        if callback is not None:
            if debug: print('emit', 'message-ack', type, json.dumps(data), "to=" + simulation_client_id,
                            "callback=" + str(callback))
            await sio.emit('message-ack', (type, json.dumps(data)), to=simulation_client_id,
                           callback=convert_response_to_message)

        else:
            if debug: print('emit', 'message', type, json.dumps(data))
            await sio.emit('message', (type, json.dumps(data)))




simulation = SimulationAPI()
gaia = GaiaAPI()