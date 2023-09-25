import asyncio
import json
import random
import datetime
from dateutil import parser

from aiohttp import web
import socketio
from cattrs import structure, unstructure

import api
from fable_agents import models
import logging

sio = socketio.AsyncServer()
app: web.Application = web.Application()
sio.attach(app)
#client_loop = asyncio.new_event_loop()
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
        choice = random.randint(0, len(msg.data['options']) - 1)
        logger.info("choice:" + msg.data['options'][choice])
        # Send back the choice.
        msg = models.Message('choose-sequence-response', {"choice": choice})
        return msg.type, json.dumps(msg.data)

    elif msg.type == 'character-status-update-tick':
        updates_raw = msg.data.get("updates", [])
        timestamp_str = msg.data.get("timestamp", '')
        # This is a hack to get around the fact that datetime.fromisoformat doesn't work for all reasonable ISO strings in python 3.10
        # See https://stackoverflow.com/questions/127803/how-do-i-parse-an-iso-8601-formatted-date which says 3.11 should fix this issue.
        #dt = datetime.datetime.fromisoformat(timestamp_str)
        dt = parser.parse(timestamp_str)
        updates = [models.StatusUpdate.from_dict(dt, json.loads(u)) for u in updates_raw]
        api.memory_datastore.add_status_updates(dt, updates)

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

        if  len(api.memory_datastore.personas) == 0:
            await api.simulation.reload_personas([], None)
            await asyncio.sleep(1)
            continue
        else:
            initiator_persona = api.memory_datastore.random_personas(1)[0]
            def handler(conversation):
                print("speaker", initiator_persona.guid)
                print("conversation", conversation)

            await api.gaia.create_conversation(initiator_persona.guid, handler)
        await asyncio.sleep(5)

async def command_interface():
    loop = asyncio.get_event_loop()
    while True:
        user_input = await loop.run_in_executor(None, input, 'Enter something: ')
        if (user_input == 'observe'):
            updates = api.memory_datastore.status_updates[api.memory_datastore.last_status_update()]
            self_update = updates.pop()
            def callback(response):
                print("CALLBACK:", self_update.guid)
                print(response)

            await api.gaia.create_observations(self_update, updates, callback)
        else:
            print(f'You entered: {user_input}')


app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(internal_tick())
    loop.create_task(command_interface())
    web.run_app(app, loop=loop)
