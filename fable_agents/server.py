import json
import random

from aiohttp import web
import socketio

sio = socketio.AsyncServer()
app = web.Application()
sio.attach(app)

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
async def echo(sid, message):
    print ('ack', message)
    return message

@sio.on('message')
async def echo(sid, message):
    print ('message', message)
    # it's probably better to not encode the message as a json string, but if we don't
    # then the client will deserialize it before we can deserialize it ourselves.
    # TODO: See if we can find an alternative to this.
    # TODO: Check the type of message first, and respond accordingly.
    try:
        data = json.loads(message)
    except json.decoder.JSONDecodeError:
        data = message
    if type(data) is dict and data['type'] == 'choose-sequence':
        choice = random.randint(0, len(data['options']) - 1)
        print('choice', data['options'][choice])
        return json.dumps({"choice": choice})
    else:
        return data


@sio.on('heartbeat')
async def heartbeat(sid):
    print('heartbeat', sid)

app.router.add_static('/static', 'static')
app.router.add_get('/', index)

if __name__ == '__main__':
    web.run_app(app)