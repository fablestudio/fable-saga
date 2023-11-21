import asyncio
import datetime
import json
import random

from dateutil import parser

from aiohttp import web
import socketio
from cattrs import structure, unstructure

from fable_agents.datastore import Datastore
from fable_agents.api import API, Resolution
from fable_agents import models, api
import logging

sio = socketio.AsyncServer()
app: web.Application = web.Application()
sio.attach(app)
# client_loop = asyncio.new_event_loop()
api.sio = sio

logger = logging.getLogger('__name__')

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
    # print ('message', message_type, message_data)
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
        current_timestamp: datetime.datetime = parser.parse(msg.data['timestamp'])
        # Get the resolution (how up-resolution to go for this request).
        resolution: str = msg.data.get('resolution', Resolution.MEDIUM)

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

            # Create a default last action in case we don't have one.
            last_action_by_persona = models.StatusUpdate(
                timestamp=current_timestamp,
                guid=persona_guid,
                sequence="idle",
                sequence_step="considering what to do next...",
                position=models.Vector3(0, 0, 0),
                location_id="",
                destination_id=""
            )

            # See if we can update the last action by the persona.
            last_update_found = Datastore.status_updates.last_update_for_persona(persona_guid)
            if last_update_found is not None:
                last_action_by_persona = last_update_found[1]

            recent_sequences = Datastore.sequence_updates.last_updates_for_persona(persona_guid, 10)
            recent_conversations = list(Datastore.conversations.get(persona_guid)[-10:])
            personas = list(Datastore.personas.personas.values())
            recent_goals = list(Datastore.recent_goals_chosen.get(persona_guid, []))

            response = await API.gaia.create_reactions(resolution, persona_guid, last_action_by_persona, last_observations,
                                                recent_sequences,
                                                Datastore.meta_affordances,
                                                recent_conversations,
                                                personas, recent_goals, current_timestamp, default_action,
                                                )
            options = response.get('options', [])
            print("OPTIONS:", response)
            Datastore.last_player_options[persona_guid] = options
            msg = models.Message('choose-sequence-response')
            msg.data['options'] = options
            msg.data['persona_guid'] = persona_guid

            return msg.type, json.dumps(msg.data)


    elif msg.type == 'character-status-update-tick':
        updates_raw = msg.data.get("updates", [])
        timestamp_str = msg.data.get("timestamp", '')
        # This is a hack to get around the fact that datetime.fromisoformat doesn't work for all reasonable ISO strings in python 3.10
        # See https://stackoverflow.com/questions/127803/how-do-i-parse-an-iso-8601-formatted-date which says 3.11 should fix this issue.
        # dt = datetime.datetime.fromisoformat(timestamp_str)
        dt = parser.parse(timestamp_str)
        updates = [models.StatusUpdate.from_dict(dt, json.loads(u)) for u in updates_raw]
        Datastore.status_updates.add_updates(dt, updates)

        for observer_guid in Datastore.personas.personas:
            persona = Datastore.personas.personas[observer_guid]
            self_update = [u for u in updates if u.guid == persona.guid][0]
            # Create observations for the observer.
            observations = await API.gaia.create_observations(self_update, updates)
            # print("CALLBACK:", self_update.guid)
            # print(observations)

    elif msg.type == 'character-conversation':
        conversation_raw = msg.data.get("conversation", None)
        timestamp_str = msg.data.get("timestamp", '')
        # This is a hack to get around the fact that datetime.fromisoformat doesn't work for all reasonable ISO strings in python 3.10
        # See https://stackoverflow.com/questions/127803/how-do-i-parse-an-iso-8601-formatted-date which says 3.11 should fix this issue.
        # dt = datetime.datetime.fromisoformat(timestamp_str)
        dt = parser.parse(timestamp_str)
        conversation = models.Conversation.from_dict(dt, json.loads(conversation_raw))
        Datastore.conversations.add(conversation)
        # TODO: Store the conversation
        # api.datastore.conversations.add_conversation(dt, conversation)

    elif msg.type == 'character-sequence-step':
        update = models.SequenceUpdate.from_dict(msg.data)
        Datastore.sequence_updates.add_updates([update])

    elif msg.type == 'affordance-state-changed':
        await API.simulation.reload_affordances([], None)

    elif msg.type == 'player-option-choice':
        persona_guid = msg.data['persona_guid']

        previous_options = Datastore.last_player_options.get(persona_guid, [])
        options = msg.data.get('options', [])
        choice_index = msg.data.get("choiceIndex")
        if not previous_options == options or not choice_index:
            return
        if choice_index < 0 or choice_index >= len(previous_options):
            return

        # Add the goal to the list of recent goals.
        # TODO: Add the timestamp of the goal at least.
        goals = Datastore.recent_goals_chosen.get(persona_guid, [])
        goals.append(previous_options[choice_index]['parameters']['goal'])
        Datastore.recent_goals_chosen[persona_guid] = goals

    else:
        logger.warning("handler not found for message type:" + msg.type + " " + str(msg.data))


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

        if len(Datastore.personas.personas) == 0:
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

        if len(Datastore.locations.locations) == 0:
            def handler():
                Datastore.locations.regenerate_hierarchy()
            await API.simulation.reload_locations(handler)
            await asyncio.sleep(1)
            continue

        await asyncio.sleep(1)


async def command_interface():
    loop = asyncio.get_event_loop()
    while True:
        user_input = await loop.run_in_executor(None, input)
        if user_input.startswith('observations'):
            args = user_input.split(' ')
            if len(args) < 2:
                print("Please specify a persona to get observations.")
                continue
            persona_guid = args[1]
            persona = Datastore.personas.personas.get(persona_guid, None)
            if persona is None:
                print(f"Persona {persona_guid} not found.")
                continue

            last_timestamp, last_observations = Datastore.observation_memory.last_observations(persona_guid)
            print(f"RECENT OBSERVATIONS for {persona_guid} - last one at :{last_timestamp}")
            print(json.dumps([api.Format.observation_event(evt) for evt in last_observations]))

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

        elif user_input.startswith('locations'):
            args = user_input.split(' ')
            if len(args) < 2:
                print("Please specify a location guid or use 'tree' to show a tree.")
                continue
            if args[1] == 'tree':
                def print_tree(node: models.LocationNode, level):
                    print('==' * level + ">", node.location.name, '?', node.location.description)
                    for child in node.children:
                        print_tree(child, level + 1)
                for node in Datastore.locations.nodes.values():
                    if node.parent is None:
                        print_tree(node, 0)

        else:
            print(f'Command not found: {user_input}')


app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(internal_tick())
    loop.create_task(command_interface())
    web.run_app(app, loop=loop)
