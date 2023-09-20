import json
import datetime
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


@define(slots=True)
class StatusUpdate:
    timestamp: datetime.datetime
    guid: str
    sequence: str
    sequence_step: str
    location: 'Vector3'
    destination: 'Vector3'

    @staticmethod
    def from_dict(timestamp: datetime.datetime, obj:dict):
        print(obj)
        params = {
            'timestamp': timestamp,
            'guid': obj['id'],
            'sequence': obj['sequence'],
            'sequence_step': obj['sequenceStep'],
            'location': Vector3.from_dict(obj['location']),
            'destination': Vector3.from_dict(obj['destination'])
        }
        return StatusUpdate(**params)
@define(slots=True)
class Vector3:
    x: float
    y: float
    z: float

    @staticmethod
    def from_dict(obj):
        if obj is None:
            return None
        params = {
            'x': obj['x'],
            'y': obj['y'],
            'z': obj['z']
        }
        return Vector3(**params)
