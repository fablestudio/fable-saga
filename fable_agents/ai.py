class Agent:

    def __init__(self, guid, location):
        self.guid = guid
        self.location = location

    def persona(self):
        #return datastore.personas[self.guid]
        return None


Actions = {
    # 'goto': {
    #     'description': "Go to a location in the world",
    #     'parameters': {
    #         'persona_guid': '<str: guid of the persona or item to go to>',
    #         'goal': '<str: goal of the movement>',
    #     },
    # },
    'converse_with': {
        'description': "Walk to another character and talk to them",
        'parameters': {
            'persona_guid': '<str: guid of the persona to converse with. You cannot talk to yourself.>',
            'topic': '<str: topic of the conversation>',
            'goal': '<str: goal of the conversation>',
        },
    },
    'wait': {
        'description': "Wait for a period of time",
        'parameters': {
            'duration': '<int: number of minutes to wait>',
            'goal': '<str: goal of the waiting>',
        },
    },
    'reflect': {
        'description': "Think about things",
        'parameters': {
            'focus': '<str: the focus of the reflection>',
            'result:': '<str: the result of the reflection (thinking to oneself). E.g., "I should go to the kitchen.">',
            'goal': '<str: goal of reflecting>',
        },
    },
    'interact': {
        'description': "Interact with an object in the world",
        'parameters': {
            'simobject_guid': 'str: The id of the sim object to interact with',
            'affordance': 'str: The name of the affordance to use when interacting',
            'goal': '<str: goal of interaction>',
        },
    },
    'continue': {
        'description': "Don't do anything new, just continue the current action",
        'parameters': {
            'goal': '<str: goal of the continuation>',
        },
    },
}







