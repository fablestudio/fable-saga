from typing import Dict
import data

class MemoryDatastore:
    personas: Dict[str, data.Persona] = {}


memory_datastore = MemoryDatastore()
