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
            'context': '<str: lots of helpful details the conversation generator can use to generate a conversation.'
                       ' It only has access to the context and the topic you provide, so be very detailed.>',
            'goal': '<str: goal of the conversation>',
        },
    },
    'wait': {
        'description': "Wait for a period of time while observing the world",
        'parameters': {
            'duration': '<int: number of minutes to wait>',
            'goal': '<str: goal of the waiting>',
        },
    },
    'reflect': {
        'description': "Think about things in order to synthesize new ideas and specific plans",
        'parameters': {
            'focus': '<str: the focus of the reflection>',
            'result:': '<str: the new specific enlightenment after reflecting.>',
            'goal': '<str: goal of reflecting>',
        },
    },
    'interact': {
        'description': "Interact with an item in the world",
        'parameters': {
            'item_guid': 'str: The id of the item to interact with',
            'interaction': 'str: The name of the interaction from the list per item.',
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







