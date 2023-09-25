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
            'persona_id': 'str: guid of the persona to go to',
        },
    },
    'converseWith': {
        'description': "Converse with a character",
        'parameters': {
            'persona_id': 'str: guid of the persona to converse with',
            'topic': 'str: topic of the conversation',
        },
    },
    'wait': {
        'description': "Wait for a period of time",
        'parameters': {
            'duration': 'int: number of seconds to wait',
        },
    },
}







