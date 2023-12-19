import datetime
from typing import List, Dict

from attr import define
from fable_saga import models


@define(slots=True)
class Persona(models.EntityInterface):
    guid: models.EntityId
    first_name: str
    last_name: str
    background: str
    appearance: str
    personality: str
    role: str

    def id(self) -> models.EntityId:
        return self.guid


@define(slots=True)
class Location(models.EntityInterface):
    guid: models.EntityId
    name: str
    description: str

    def id(self) -> models.EntityId:
        return self.guid


@define(slots=True)
class InteractableObject(models.EntityInterface):
    guid: models.EntityId
    location: models.EntityId
    affordances: List[str]

    def id(self) -> models.EntityId:
        return self.guid


@define(slots=True)
class Skill(models.EntityInterface):
    guid: models.EntityId
    description: str
    parameters: Dict[str, str]

    def id(self) -> models.EntityId:
        return self.guid


@define(slots=True)
class Memory:
    summary: str
    timestamp: datetime.datetime
