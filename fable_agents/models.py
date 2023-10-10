import json
import datetime
from typing import Any, Optional

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
class Location:
    vector3: 'Vector3'
    place: str
    description: str

    @staticmethod
    def from_dict(obj):
        if obj is None:
            return None
        params = {
            'vector3': Vector3.from_dict(obj['vector3']),
            'place': obj['place'],
            'description': obj['description']
        }
        return Location(**params)


@define(slots=True)
class StatusUpdate:
    timestamp: datetime.datetime
    guid: str
    sequence: str
    sequence_step: str
    location: Location
    destination: Location

    @staticmethod
    def from_dict(timestamp: datetime.datetime, obj: dict):
        params = {
            'timestamp': timestamp,
            'guid': obj['id'],
            'sequence': obj['sequence'],
            'sequence_step': obj['sequenceStep'],
            'location': Location.from_dict(obj['location']),
            'destination': Location.from_dict(obj['destination'])
        }
        return StatusUpdate(**params)


@define(slots=True)
class SequenceStep:
    timestamp: datetime.datetime
    guid: str
    sequence: str
    starting_step: str
    completed_step: str
    completed_step_duration: float
    interrupted: bool

    @staticmethod
    def from_dict(timestamp: datetime.datetime, obj: dict):
        params = {
            'timestamp': timestamp,
            'guid': obj['npcId'],
            'sequence': obj['sequenceName'],
            'starting_step': obj['startingStep'],
            'completed_step': obj['completedStep'],
            'completed_step_duration': obj['completedStepDurationMinutes'],
            'interrupted': obj['interrupted']
        }
        return SequenceStep(**params)


@define(slots=True)
class Conversation:
    timestamp: datetime.datetime
    turns: ['ConversationTurn']

    @staticmethod
    def from_dict(timestamp: datetime.datetime, obj: dict):

        params = {
            'timestamp': timestamp,
            'turns': [ConversationTurn.from_dict(turnData) for turnData in obj['turns']],
        }
        return Conversation(**params)


@define(slots=True)
class ConversationTurn:
    guid: str
    dialogue: str

    @staticmethod
    def from_dict(obj: dict):
        params = {
            'guid': obj['npcId'],
            'dialogue': obj['dialogue'],
        }
        return ConversationTurn(**params)


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
            'distance': Vector3.distance(update.location.vector3,  observer_update.location.vector3),
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


@define(slots=True)
class SequenceUpdate:
    """
    {'sequence': '{"npcId":"henry_jenkins","sequenceName":"Play Cards: ",
        "startingStep":"Ready to play","completedStep":"Go to game table",
        "completedStepDurationMinutes":14.584982872009278,"interrupted":false}',
        'timestamp': '2000-01-01T08:15:44.021'
    }
    """
    timestamp: datetime.datetime
    persona_guid: str
    action: str
    action_step_started: Optional[str]
    action_step_completed: Optional[str]
    interrupted: bool

    @staticmethod
    def from_dict(obj):
        seq = json.loads(obj['sequence'])

        params = {
            'timestamp': obj['timestamp'],
            'persona_guid': seq['npcId'],
            'action': seq['sequenceName'],
            'action_step_started': seq['startingStep'],
            'action_step_completed': seq['completedStep'],
            'interrupted': seq['interrupted']
        }
        return SequenceUpdate(**params)