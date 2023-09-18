import asyncio
import json
import random

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
    print ('message', message_type, message_data)
    # it's probably better to not encode the message as a json string, but if we don't
    # then the client will deserialize it before we can deserialize it ourselves.
    # TODO: See if we can find an alternative to this.
    # TODO: Check the type of message first, and respond accordingly.
    try:
        msg_data = json.loads(message_data)

    except json.decoder.JSONDecodeError:
        msg_data = message_data
    msg = models.Message(message_type, msg_data)
    if msg.type == 'choose-sequence':
        choice = random.randint(0, len(msg.data['options']) - 1)
        print('choice', msg.data['options'][choice])
        # Send back the choice.
        msg = models.Message('choose-sequence-response', {"choice": choice})
        return msg.type, json.dumps(msg.data)
    else:
        return msg.type, json.dumps(msg.data)


@sio.on('heartbeat')
async def heartbeat(sid):
    print('heartbeat', sid)

async def sync_personas():
    """
    Sync the personas with the server.
    """
    while True:
        print("syncing personas 1")
        await asyncio.sleep(5)
        print("syncing personas 2")
        await simulation.reload_personas([], lambda: print('personas loaded 3'))


app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    loop.create_task(sync_personas())
    web.run_app(app, loop=loop)