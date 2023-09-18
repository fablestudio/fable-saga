import json
from dataclasses import dataclass
from typing import List, Callable, Optional, Any
import models
from datastore import memory_datastore
import socketio


sio: socketio.AsyncServer = None
simulation_client_id: Optional[str] = None

debug=True


async def send(type: str, data: dict, callback: Callable[[models.Message], None]):
    """
    Send a request to the server.
    :param type: type of request.
    :param data: data to send with the request.
    :param callback: function to call when the server responds.
    """

    def convert_response_to_message(response_type:str, response_data:str):
        response_message = models.Message(response_type, json.loads(response_data))
        callback(response_message)

    if callback is not None:
        if debug: print('emit', 'message-ack', [type, json.dumps(data)], "to=" + simulation_client_id, "callback=" + str(callback))
        await sio.emit('message-ack', (type, json.dumps(data)), to=simulation_client_id, callback=convert_response_to_message)

    else:
        if debug: print('emit', 'message', [type, json.dumps(data)])
        await sio.emit('message', (type, json.dumps(data)))


class GaiaAPI:

    async def create_conversation(self, persona_guids: List[str], on_complete: Callable[[models.Message], None]):
        """
        Create a new conversation.
        :param persona_guids: unique identifiers for the personas.
        :param callback: function to call when the conversation is created.
        """
        pass






class SimulationAPI:

    async def reload_personas(self, guids: List[str], on_complete: Callable[[], None]):
        """
        Load persona's current state (i.e. description and memory).
        :param guids: unique identifiers for the persona.
        :param callback: function to call when the personas are loaded.
        """

        def convert_to_personas(response: models.Message):
            print("RESPONSE", response)
            # for d in response:
            #     memory_datastore.personas[d['guid']] = data.Persona(**d)

            # TODO: This should return when this process is complete.
            on_complete()

        # Note: Server callback only works with a specific client id.
        await send('request-personas', {'guids': []}, callback=convert_to_personas)
