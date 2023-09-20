import json
from dataclasses import dataclass
import random
from typing import List, Callable, Optional, Any
import models
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


class GaiaAPI:

    async def create_conversation(self, persona_guid:str, on_complete: Callable[[models.Message], None]):
        """
        Create a new conversation.
        :param persona_guids: unique identifiers for the personas.
        :param callback: function to call when the conversation is created.
        """
        initiator_persona = memory_datastore.personas.get(persona_guid, None)
        if initiator_persona is None:
            print('Error: persona not found.')
            if on_complete is not None:
                on_complete(models.Message('error', {'error': 'persona not found.'}))
            return

        def format_persona(persona: models.Persona):
            return {
                'ID': persona.guid,
                'FirstName': persona.first_name,
                'LastName': persona.last_name,
                'Description': persona.description,
                'Summary': persona.summary,
                'BackStory': persona.backstory
            }

        # Create a list of personas to send to the server as options.
        options = []

        # TODO: Using all personas results in too much text.
        # pick 5 random personas
        items = list(memory_datastore.personas.items())
        random.shuffle(items)
        for guid, persona_option in items[:5]:
            options.append(format_persona(persona_option))

        prompt = load_prompt("prompt_templates/conversation_v1.yaml")
        llm = ChatOpenAI(temperature=0.9, model_name="gpt-3.5-turbo-0613")
        chain = LLMChain(llm=llm, prompt=prompt)
        resp = await chain.arun(self_description=json.dumps(format_persona(initiator_persona)), options=json.dumps(options))
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

        def convert_to_personas(response: models.Message):
            print("RESPONSE", response)
            if (response.type != 'request-personas-response'):
                print('Error: expected personas response.', response)
                if on_complete is not None:
                    on_complete()
                return
            for json_rep in response.data['personas']:
                persona = models.Persona.from_json(json_rep)
                memory_datastore.personas[persona.guid] = persona

            # TODO: This should return when this process is complete.
            if on_complete is not None:
                on_complete()

        # Note: Server callback only works with a specific client id.
        await self.send('request-personas', {'guids': guids}, callback=convert_to_personas)

    async def send(self, type: str, data: dict, callback: Callable[[models.Message], None]):
        """
        Send a request to the server.
        :param type: type of request.
        :param data: data to send with the request.
        :param callback: function to call when the server responds.
        """

        def convert_response_to_message(response_type: str, response_data: str):
            response_message = models.Message(response_type, json.loads(response_data))
            callback(response_message)

        if simulation_client_id is None:
            print('Error: simulation_client_id is None.')
            if callback is not None:
                callback(models.Message('error', {'error': 'simulation_client_id is None.'}))
            return
        if callback is not None:
            if debug: print('emit', 'message-ack', type, json.dumps(data), "to=" + simulation_client_id,
                            "callback=" + str(callback))
            await sio.emit('message-ack', (type, json.dumps(data)), to=simulation_client_id,
                           callback=convert_response_to_message)

        else:
            if debug: print('emit', 'message', type, json.dumps(data))
            await sio.emit('message', (type, json.dumps(data)))

