from typing import Dict, List, Tuple, Optional, Any
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


class ObservationMemory:

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

    def get_or_create(self, guid) -> models.Persona:
        if guid not in self.personas:
            self.personas[guid] = models.Persona(guid)
        return self.personas[guid]

    def random_personas(self, n):
        keys = list(self.personas.keys())
        random.shuffle(keys)
        return [self.personas[k] for k in keys[:n]]


class Locations:

    def __init__(self):
        self.locations: Dict[str, models.Location] = {}
        self.nodes: Dict[str, models.LocationNode] = {}

    def regenerate_hierarchy(self):
        self.nodes.clear()
        need_parents: List[models.LocationNode] = []

        # First, create all the nodes.
        for location in self.locations.values():
            node = models.LocationNode(location, None, [])
            self.nodes[location.guid] = node
            if location.parent_guid is not None and location.parent_guid != '':
                need_parents.append(node)
        # Now, add the parents.
        while len(need_parents) > 0:
            node = need_parents.pop()
            parent = self.nodes.get(node.location.parent_guid, None)
            if parent is None:
                # This parent hasn't been created yet, throw an error.
                raise Exception(f"Parent {node.location.parent_guid} not found for location {node.location.guid}")
            # Add the parent to the node.
            node.parent = parent
            # Add the node to the parent's children.
            parent.children.append(node)


class MetaAffordances:

    def __init__(self):
        self.affordances: Dict[str, models.MetaAffordanceProvider] = {}

    def get(self, key) -> models.MetaAffordanceProvider:
        return self.affordances[key]

    def simobjects(self) -> List[str]:
        return list(self.affordances.keys())


class StatusUpdates:

    def __init__(self):
        self.status_updates: Dict[datetime.datetime, List[models.StatusUpdate]] = {}

    def add_updates(self, timestamp, updates: List[models.StatusUpdate]):
        # for now, just store the updates. Later we will do something with them.
        if timestamp not in self.status_updates:
            self.status_updates[timestamp] = []
        self.status_updates[timestamp].extend(updates)

    def last_updates(self) -> Tuple[Optional[datetime.datetime], List[models.StatusUpdate]]:
        if len(self.status_updates) == 0:
            return None, []
        timestamp = list(self.status_updates)[-1]
        return timestamp, self.status_updates[timestamp]

    def last_update_for_persona(self, persona_guid) -> Optional[Tuple[datetime.datetime, models.StatusUpdate]]:
        timestamp, status_updates = self.last_updates()
        if timestamp is None:
            return None
        updates = [u for u in status_updates if u.guid == persona_guid]
        return timestamp, updates[0]


class SequenceUpdates:

    def __init__(self):
        self.sequence_updates: Dict[str, List[models.SequenceUpdate]] = {}

    def add_updates(self, updates: List[models.SequenceUpdate]):
        for update in updates:
            persona_id = update.persona_guid
            persona_updates = self.sequence_updates.get(persona_id, [])
            persona_updates.append(update)
            self.sequence_updates[persona_id] = persona_updates

    def last_updates_for_persona(self, persona_id, n) -> List[models.SequenceUpdate]:
        return self.sequence_updates.get(persona_id, [])[-n:]


class Memories:

    def __init__(self):
        self._memories: Dict[str, List[models.Memory]] = {}

    def get(self, persona_id: models.EntityId) -> List[models.Memory]:
        if persona_id not in self._memories:
            self._memories[persona_id] = []
        return self._memories.get(persona_id)

    def add(self, persona_id, memory: models.Memory):
        self.get(persona_id).append(memory)

    def reset(self):
        self._memories.clear()

    # def knowledge_graph(self, persona_id) -> models.KnowledgeGraph:
    #     memories = self._memories.get(persona_id, [])
    #     knowledge_graph = models.KnowledgeGraph()
    #     for memory in memories:
    #         knowledge_graph.add_memory(memory)



class ConversationMemory:

    conversations: Dict[str, List[models.Conversation]] = {}

    def add(self, conversation: models.Conversation) -> None:
        """
        Add a conversation to the memory. Indexed by speakers guids.
        :param conversation:
        """
        guids = set([turn.guid for turn in conversation.turns])
        for guid in guids:
            conversations = self.conversations.get(guid, [])
            conversations.append(conversation)
            self.conversations[guid] = conversations

    def get(self, guid) -> List[models.Conversation]:
        return self.conversations.get(guid, [])


class Datastore:
    conversations: ConversationMemory = ConversationMemory()
    observation_memory: ObservationMemory = ObservationMemory()
    personas: Personas = Personas()
    meta_affordances: MetaAffordances = MetaAffordances()
    status_updates: StatusUpdates = StatusUpdates()
    sequence_updates: SequenceUpdates = SequenceUpdates()
    memory_vectors: MemoryVectors = MemoryVectors()
    locations: Locations = Locations()
    last_player_options: Dict[str, Optional[List[Dict[str, Any]]]] = {}
    recent_goals_chosen: Dict[str, List[str]] = {}
    memories: Memories = Memories()
