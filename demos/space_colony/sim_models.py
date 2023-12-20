import datetime
from typing import List, Dict
from attr import define
from fable_saga import EntityId, EntityInterface


@define(slots=True)
class Persona(EntityInterface):
    guid: EntityId
    first_name: str
    last_name: str
    background: str
    appearance: str
    personality: str
    role: str

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
class Skill(EntityInterface):
    guid: EntityId
    description: str
    parameters: Dict[str, str]

    def id(self) -> EntityId:
        return self.guid


@define(slots=True)
class Memory:
    summary: str
    timestamp: datetime.datetime
