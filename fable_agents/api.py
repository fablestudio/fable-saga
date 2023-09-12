from dataclasses import dataclass
from typing import List, Callable
import data
from datastore import memory_datastore
import socketio


sio: socketio.AsyncServer = None


class SimulationAPI:

    async def reload_personas(self, guids: List[str], callback: Callable[[], None]):
        """
        Load persona's current state (i.e. description and memory).
        :param guids: unique identifiers for the persona.
        :param callback: function to call when the personas are loaded.
        """

        def convert_to_personas(response):

            for d in response:
                memory_datastore.personas[d['guid']] = types.Persona(**d)
            callback()

        await sio.emit('persona', guids, callback=convert_to_personas)