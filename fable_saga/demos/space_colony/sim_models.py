import datetime
from abc import ABC, abstractmethod
from typing import List, Dict, Any

from attr import define

EntityId = str


class EntityInterface(ABC):
    @abstractmethod
    def id(self) -> EntityId:
        pass


@define(slots=True)
class Persona(EntityInterface):
    guid: EntityId
    first_name: str
    last_name: str
    background: str
    appearance: str
    personality: str
    role: str
    job: str

    def id(self) -> EntityId:
        return self.guid


@define(slots=True)
class Location(EntityInterface):
    guid: EntityId
    name: str
    description: str

    def id(self) -> EntityId:
        return self.guid


@define(slots=True)
class InteractableObject(EntityInterface):
    guid: EntityId
    location: EntityId
    affordances: List[str]

    def id(self) -> EntityId:
        return self.guid


@define(slots=True)
class Memory:
    summary: str
    timestamp: datetime.datetime
    metadata: Dict[str, Any] = {}
