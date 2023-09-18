from typing import Any

from attrs import define
from cattrs import structure, unstructure


@define(slots=True)
class Persona:
    guid: str
    name_id: str
    description: str
    backstory: str


@define(slots=True)
class Message:
    type: str
    data: dict
