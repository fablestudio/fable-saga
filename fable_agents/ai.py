from datastore import memory_datastore
from fable_agents import api
from models import Vector3
class Agent:

    def __init__(self, guid, location):
        self.guid = guid
        self.location = location

    def persona(self):
        return memory_datastore.personas[self.guid]







