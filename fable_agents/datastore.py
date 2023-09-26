from typing import Dict, List, Tuple, Optional
import models
import random
import datetime
from langchain.memory import VectorStoreRetrieverMemory
from langchain.embeddings.openai import OpenAIEmbeddings
import faiss

from langchain.docstore import InMemoryDocstore
from langchain.vectorstores import FAISS


class MemoryVectors:

    def __init__(self):
        # Set up the vector store
        embedding_size = 1536  # Dimensions of the OpenAIEmbeddings
        index = faiss.IndexFlatL2(embedding_size)
        embedding_fn = OpenAIEmbeddings().embed_query
        vectorstore = FAISS(embedding_fn, index, InMemoryDocstore({}), {})
        retriever = vectorstore.as_retriever(search_kwargs={'k': 6, 'lambda_mult': 0.25}, )

        # Set up the memory
        self.memory_vectors = VectorStoreRetrieverMemory(retriever=retriever)


class ObservationMemory():

    def __init__(self):
        self.observation_memory: Dict[str, Dict[datetime, List[models.ObservationEvent]]] = {}

    def set_observations(self, observer_guid, timestamp, observations: List[models.ObservationEvent]):
        if observer_guid not in self.observation_memory:
            self.observation_memory[observer_guid] = {}
        self.observation_memory[observer_guid][timestamp] = observations

    def last_observations(self, observer_guid) -> Tuple[Optional[datetime.datetime], List[models.ObservationEvent]]:
        persona_observations = self.observation_memory.get(observer_guid, None)
        if persona_observations is None:
            return None, []
        timestamps = list(persona_observations.keys())
        if len(timestamps) == 0:
            return None, []
        timestamp = timestamps[-1]
        return timestamp, self.observation_memory[observer_guid][timestamp]


class Personas:

    def __init__(self):
        self.personas: Dict[str, models.Persona] = {}

    def random_personas(self, n):
        keys = list(self.personas.keys())
        random.shuffle(keys)
        return [self.personas[k] for k in keys[:n]]


class StatusUpdates:

    def __init__(self):
        self.status_updates: Dict[datetime.datetime, List[models.StatusUpdate]] = {}

    def add_updates(self, timestamp, updates: List[models.StatusUpdate]):
        # for now, just store the updates. Later we will do something with them.
        if timestamp not in self.status_updates:
            self.status_updates[timestamp] = []
        self.status_updates[timestamp].extend(updates)

    def last_updates(self) -> Tuple[datetime.datetime, List[models.StatusUpdate]]:
        timestamp = list(self.status_updates)[-1]
        return timestamp, self.status_updates[timestamp]

    def last_update_for_persona(self, persona_guid) -> Tuple[datetime.datetime, models.StatusUpdate]:
        timestamp, status_updates = self.last_updates()
        updates = [u for u in self.status_updates[timestamp] if u.guid == persona_guid]
        return timestamp, updates[0]


observation_memory = ObservationMemory()
personas = Personas()
status_updates = StatusUpdates()
memory_vectors = MemoryVectors()
