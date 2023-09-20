from typing import Dict, List
import models
import random
import datetime

class MemoryDatastore:
    personas: Dict[str, models.Persona] = {}
    status_updates: Dict[datetime.datetime, List[models.StatusUpdate]] = {}

    def random_personas(self, n):
        keys = list(self.personas.keys())
        random.shuffle(keys)
        return [self.personas[k] for k in keys[:n]]

    def add_status_updates(self, timestamp, updates: List[models.StatusUpdate]):
        # for now, just store the updates. Later we will do something with them.
        if timestamp not in self.status_updates:
            self.status_updates[timestamp] = []
        self.status_updates[timestamp].extend(updates)


memory_datastore = MemoryDatastore()
