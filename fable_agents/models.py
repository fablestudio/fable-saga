import json
import datetime
from typing import Any, Optional, List

from attrs import define
from cattrs import structure, unstructure
from dateutil import parser



class World:
    last_update: datetime.datetime


@define(slots=True)
class Persona:
    guid: str
    name_id: str
    first_name: str
    last_name: str
    description: str
    summary: str
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
class SimObject:
    guid: str
    display_name: str
    description: str

    @staticmethod
    def from_dict(obj):
        params = {
            'guid': obj['id'],
            'display_name': obj['displayName'],
            'description': obj['description'],
        }
        return SimObject(**params)


@define(slots=True)
class MetaAffordanceProvider:
    sim_object: SimObject
    affordances: [str]
    locked: bool

    @staticmethod
    def from_json(json_string):
        obj = json.loads(json_string)
        params = {
            'sim_object': SimObject.from_dict(obj['simObject']),
            'affordances': obj['affordanceNames'],
            'locked': obj['locked'],
        }
        return MetaAffordanceProvider(**params)


@define(slots=True)
class Message:
    type: str
    data: dict


@define(slots=True)
class Location:
    guid: str
    name: str
    description: str
    parent_guid: str
    center: 'Vector3'
    extents: 'Vector3'
    center_floor_position: 'Vector3'

    @staticmethod
    def from_json(json_string: str):
        obj = json.loads(json_string)
        return Location.from_dict(obj)

    @staticmethod
    def from_dict(obj):
        if obj is None:
            return None
        params = {
            'guid': obj['id'],
            'name': obj['name'],
            'description': obj['description'],
            'parent_guid': obj['parentId'],
            'center': Vector3.from_dict(obj['center']),
            'extents': Vector3.from_dict(obj['extents']),
            'center_floor_position': Vector3.from_dict(obj['centerFloorPosition'])
        }
        return Location(**params)


@define(slots=True)
class StatusUpdate:
    timestamp: datetime.datetime
    guid: str
    sequence: str
    sequence_step: str
    position: 'Vector3'
    location_id: str
    destination_id: str

    @staticmethod
    def from_dict(timestamp: datetime.datetime, obj: dict):
        params = {
            'timestamp': timestamp,
            'guid': obj['id'],
            'sequence': obj['sequence'],
            'sequence_step': obj['sequenceStep'],
            'position': Vector3.from_dict(obj['position']),
            'location_id': obj['locationId'],
            'destination_id': obj['destinationId']
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
    turns: List['ConversationTurn']

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
            'distance': Vector3.distance(update.position, observer_update.position),
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
            'timestamp': parser.parse(obj['timestamp']),
            'persona_guid': seq['npcId'],
            'action': seq['sequenceName'],
            'action_step_started': seq['startingStep'],
            'action_step_completed': seq['completedStep'],
            'interrupted': seq['interrupted']
        }
        return SequenceUpdate(**params)

@define(slots=True)
class LocationNode:
    location: Location
    parent: Optional['LocationNode']
    children: List['LocationNode']