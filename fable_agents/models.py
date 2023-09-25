import json
import datetime
from typing import Any

from attrs import define
from cattrs import structure, unstructure


class World:
    last_update: datetime.datetime


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

    @staticmethod
    def distance(v1, v2):
        return ((v1.x - v2.x)**2 + (v1.y - v2.y)**2 + (v1.z - v2.z)**2)**0.5

@define(slots=True)
class ObservationEvent:
    timestamp: datetime.datetime
    observer_guid: str
    persona_guid: str
    action: str
    action_step: str
    distance: float
    summary: str
    importance: int

    @staticmethod
    def from_status_update(update: StatusUpdate, observer_update: StatusUpdate) -> 'ObservationEvent':
        params = {
            'observer_guid': observer_update.guid,
            'timestamp': update.timestamp,
            'persona_guid': update.guid,
            'action': update.sequence,
            'action_step': update.sequence_step,
            'distance': Vector3.distance(update.location,  observer_update.location),
            'summary': '',
            'importance': 0
        }
        return ObservationEvent(**params)

    @staticmethod
    def from_dict(obj) -> 'ObservationEvent':
        params = {
            'observer_guid': obj['observer_guid'],
            'timestamp': obj['timestamp'],
            'persona_guid': obj['persona_guid'],
            'action': obj['action'],
            'action_step': obj['action_step'],
            'distance': obj['distance'],
            'summary': obj.get('summary', ''),
            'importance': obj.get('importance', 0)
        }
        return ObservationEvent(**params)