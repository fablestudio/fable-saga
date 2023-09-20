import asyncio
import json
import random
import datetime

from aiohttp import web
import socketio
from cattrs import structure, unstructure

import api
from fable_agents import models

sio = socketio.AsyncServer()
app: web.Application = web.Application()
sio.attach(app)
#client_loop = asyncio.new_event_loop()
api.sio = sio
simulation = api.SimulationAPI()
gaia = api.GaiaAPI()

async def index(request):
    """Serve the client-side application."""
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.event
def connect(sid, environ):
    api.simulation_client_id = sid
    print("connect ", sid)

@sio.event
async def chat_message(sid, data):
    print("message ", data)

@sio.event
def disconnect(sid):
    print('disconnect ', sid)

@sio.on('echo')
async def echo(sid, message):
    print ('echo', message)
    await sio.emit('echo', message)

@sio.on('ack')
async def ack(sid, type, data):
    print ('ack', type, data)
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
        print('choice', msg.data['options'][choice])
        # Send back the choice.
        msg = models.Message('choose-sequence-response', {"choice": choice})
        return msg.type, json.dumps(msg.data)
    elif msg.type == 'character-status-update-tick':
        updates_raw = msg.data.get("updates", [])
        timestamp_str = msg.data.get("timestamp", '')
        dt = datetime.datetime.fromisoformat(timestamp_str)
        updates = [models.StatusUpdate.from_dict(dt, json.loads(u)) for u in updates_raw]
        api.memory_datastore.add_status_updates(dt, updates)
    else:
        print("handler not found for message type", msg.type)


@sio.on('heartbeat')
async def heartbeat(sid):
    print('heartbeat', sid)

async def internal_tick():
    """
    Sync the personas with the server.
    """
    while True:
        if api.simulation_client_id is None:
            await asyncio.sleep(1)
            continue

        if  len(api.memory_datastore.personas) == 0:
            await simulation.reload_personas([], None)
            await asyncio.sleep(1)
            continue
        else:
            initiator_persona = api.memory_datastore.random_personas(1)[0]
            def handler(conversation):
                print("speaker", initiator_persona.guid)
                print("conversation", conversation)

            await gaia.create_conversation(initiator_persona.guid, handler)
        await asyncio.sleep(5)


app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(internal_tick())
    web.run_app(app, loop=loop)