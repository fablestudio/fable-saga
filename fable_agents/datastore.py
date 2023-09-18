from typing import Dict
import models

class MemoryDatastore:
    personas: Dict[str, models.Persona] = {}


memory_datastore = MemoryDatastore()
