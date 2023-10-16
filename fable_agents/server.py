import asyncio
import json
import random
from dateutil import parser

from aiohttp import web
import socketio
from cattrs import structure, unstructure

from fable_agents.datastore import Datastore
from fable_agents.api import API
from fable_agents import models, api
import logging

sio = socketio.AsyncServer()
app: web.Application = web.Application()
sio.attach(app)
#client_loop = asyncio.new_event_loop()
api.sio = sio

logger = logging.getLogger('__name__')

auto_observer_guids = ['wyatt_cooper']

async def index(request):
    """Serve the client-side application."""
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.event
def connect(sid, environ):
    api.simulation_client_id = sid
    logger.info("connect:" + sid)

@sio.event
def disconnect(sid):
    logger.info("disconnect:" + sid)

@sio.on('echo')
async def echo(sid, message):
    logger.info("echo:" + message)
    await sio.emit('echo', message)

@sio.on('ack')
async def ack(sid, type, data):
    logger.info("ack:" + type + " " + data)
    return type, data

@sio.on('message')
async def message(sid, message_type, message_data):
    #print ('message', message_type, message_data)
    # it's probably better to not encode the message as a json string, but if we don't
    # then the client will deserialize it before we can deserialize it ourselves.
    # TODO: See if we can find an alternative to this.
    # TODO: Check the type of message first, and respond accordingly.
    try:
        parsed_data = json.loads(message_data)

    except json.decoder.JSONDecodeError:
        parsed_data = message_data
    msg = models.Message(message_type, parsed_data)

    if msg.type == 'choose-sequence':
        use_random = False

        if use_random:
            # Choose a random option.
            choice = random.randint(0, len(msg.data['options']) - 1)
            logger.info("choice:" + msg.data['options'][choice])
            # Send back the choice.
            msg = models.Message('choose-sequence-response', {"choice": choice})
            return msg.type, json.dumps(msg.data)
        else:
            # Generate one or more options.
            persona_guid = msg.data['persona_guid']
            if persona_guid not in Datastore.personas.personas:
                print(f"Persona {persona_guid} not found.")
                return
            last_ts, last_observations = Datastore.observation_memory.last_observations(persona_guid)
            last_ts, last_update = Datastore.status_updates.last_update_for_persona(persona_guid)
            recent_sequences = Datastore.sequence_updates.last_updates_for_persona(persona_guid, 10)
            recent_conversations = Datastore.conversations.get(persona_guid)[-10:]

            options = await API.gaia.create_reactions(last_update, last_observations, recent_sequences,
                                                      Datastore.meta_affordances, recent_conversations,ignore_continue=True)
            print("OPTIONS:", options)
            # options = [{'action': 'interact', 'parameters': {'simobject_guid': 'Bank', 'affordance': 'Rob Bank'}}]
            msg = models.Message('choose-sequence-response', {"options": options})
            return msg.type, json.dumps(msg.data)


    elif msg.type == 'character-status-update-tick':
        updates_raw = msg.data.get("updates", [])
        timestamp_str = msg.data.get("timestamp", '')
        # This is a hack to get around the fact that datetime.fromisoformat doesn't work for all reasonable ISO strings in python 3.10
        # See https://stackoverflow.com/questions/127803/how-do-i-parse-an-iso-8601-formatted-date which says 3.11 should fix this issue.
        #dt = datetime.datetime.fromisoformat(timestamp_str)
        dt = parser.parse(timestamp_str)
        updates = [models.StatusUpdate.from_dict(dt, json.loads(u)) for u in updates_raw]
        Datastore.status_updates.add_updates(dt, updates)

        for observer_guid in auto_observer_guids:
            persona = Datastore.personas.personas[observer_guid]
            self_update = [u for u in updates if u.guid == persona.guid][0]

            # Create observations for the observer.
            observations = await API.gaia.create_observations(self_update, updates)
            #print("CALLBACK:", self_update.guid)
            #print(observations)

    elif msg.type == 'character-conversation':
        conversation_raw = msg.data.get("conversation", None)
        timestamp_str = msg.data.get("timestamp", '')
        # This is a hack to get around the fact that datetime.fromisoformat doesn't work for all reasonable ISO strings in python 3.10
        # See https://stackoverflow.com/questions/127803/how-do-i-parse-an-iso-8601-formatted-date which says 3.11 should fix this issue.
        #dt = datetime.datetime.fromisoformat(timestamp_str)
        dt = parser.parse(timestamp_str)
        conversation = models.Conversation.from_dict(dt, json.loads(conversation_raw))
        Datastore.conversations.add(conversation)
        # TODO: Store the conversation
        # api.datastore.conversations.add_conversation(dt, conversation)

    elif msg.type == 'character-sequence-step':
        # Ignore sequences that are not for the auto observer.
        update = models.SequenceUpdate.from_dict(msg.data)
        if update.persona_guid not in auto_observer_guids:
            return
        Datastore.sequence_updates.add_updates([update])

    elif msg.type == 'affordance-state-changed':
        await API.simulation.reload_affordances([], None)

    else:
        logger.warning("handler not found for message type:" + msg.type)


@sio.on('heartbeat')
async def heartbeat(sid):
    logger.info('heartbeat:' + sid)

async def internal_tick():
    """
    Sync the personas with the server.
    """
    while True:
        if api.simulation_client_id is None:
            await asyncio.sleep(1)
            continue

        if  len(Datastore.personas.personas) == 0:
            await API.simulation.reload_personas([], None)
            await asyncio.sleep(1)
            continue
        else:
            pass
            # initiator_persona = api.datastore.personas.random_personas(1)[0]
            # def handler(conversation):
            #     print("speaker", initiator_persona.guid)
            #     print("conversation", conversation)
            #
            # await api.gaia.create_conversation(initiator_persona.guid, handler)

        if len(Datastore.meta_affordances.affordances) == 0:
            await API.simulation.reload_affordances([], None)
            await asyncio.sleep(1)
            continue

        await asyncio.sleep(5)

async def command_interface():
    loop = asyncio.get_event_loop()
    while True:
        user_input = await loop.run_in_executor(None, input)
        if user_input.startswith('observe'):
            args = user_input.split(' ')
            if len(args) < 2:
                print("Please specify a persona to observe.")
                continue
            if args[1] not in Datastore.personas:
                print(f"Persona {args[1]} not found.")
                continue

            persona = Datastore.personas[args[1]]
            updates = Datastore.status_updates[Datastore.status_updates.last_status_update()]
            self_update = [u for u in updates if u.guid == persona.guid][0]

            await API.gaia.create_observations(self_update, updates)
        elif user_input.startswith('recall'):
            context = user_input.replace('recall ', '')
            memories = Datastore.memory_vectors.memory_vectors.load_memory_variables({'context': context})
            print("RECALL:", memories)

        elif user_input.startswith('react'):
            args = user_input.split(' ')
            if len(args) < 2:
                print("Please specify a persona to react.")
                continue
            reactor_guid = args[1]
            if reactor_guid not in Datastore.personas.personas:
                print(f"Persona {args[1]} not found.")
                continue
            last_ts, last_observations = Datastore.observation_memory.last_observations(reactor_guid)
            last_ts, last_update = Datastore.status_updates.last_update_for_persona(reactor_guid)
            reactions = await API.gaia.create_reactions(last_update, last_observations)
            print("REACTIONS:", reactions)

        else:
            print(f'Command not found: {user_input}')



app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(internal_tick())
    loop.create_task(command_interface())
    web.run_app(app, loop=loop)
