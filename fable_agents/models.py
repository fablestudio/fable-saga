import json
from typing import Any

from attrs import define
from cattrs import structure, unstructure


@define(slots=True)
class Persona:
    guid: str
    name_id: str
    first_name: str
    last_name: str
    description: str
    summary:str
    backstory: str

    @staticmethod
    def from_json(json_string):
        obj = json.loads(json_string)
        params = {
            'guid': obj['ID'],
            'name_id': obj['ID'],
            'first_name':obj['FirstName'],
            'last_name': obj['LastName'],
            'description': obj['Description'],
            'summary': obj['Summary'],
            'backstory': obj['BackStory']
        }
        return Persona(**params)


@define(slots=True)
class Message:
    type: str
    data: dict
