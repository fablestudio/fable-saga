class Agent:

    def __init__(self, guid, location):
        self.guid = guid
        self.location = location

    def persona(self):
        #return datastore.personas[self.guid]
        return None


Actions = {
    'goto': {
        'description': "Go to a character's location",
        'parameters': {
            'persona_guid': 'str: guid of the persona to go to',
            'goal': 'str: goal of the movement',
        },
    },
    'converse_with': {
        'description': "Converse with a character",
        'parameters': {
            'persona_guid': 'str: guid of the persona to converse with',
            'topic': 'str: topic of the conversation',
            'goal': 'str: goal of the conversation',
        },
    },
    'wait': {
        'description': "Wait for a period of time",
        'parameters': {
            'duration': 'int: number of seconds to wait',
            'goal': 'str: goal of the waiting',
        },
    },
    'continue': {
        'description': "Don't do anything new, just continue the current action",
        'parameters': {
            'goal': 'str: goal of the continuation',
        },
    },
}







