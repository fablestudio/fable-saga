import json
import random

from aiohttp import web
import socketio
from cattrs import structure, unstructure

import api
from fable_agents import data

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

api.sio = sio

async def index(request):
    """Serve the client-side application."""
    with open('index.html') as f:
        return web.Response(text=f.read(), content_type='text/html')

@sio.event
def connect(sid, environ):
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
    msg = data.Message(message_type, msg_data)
    if msg.type == 'choose-sequence':
        choice = random.randint(0, len(msg.data['options']) - 1)
        print('choice', msg.data['options'][choice])
        # Send back the choice.
        msg = data.Message('choose-sequence-response', json.dumps({"choice": choice}))
        return msg.type, msg.data
    else:
        return msg.type, msg.data


@sio.on('heartbeat')
async def heartbeat(sid):
    print('heartbeat', sid)

app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app)