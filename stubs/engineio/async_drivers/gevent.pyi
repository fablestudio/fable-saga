from _typeshed import Incomplete
from engineio.async_drivers._websocket_wsgi import SimpleWebSocketWSGI as SimpleWebSocketWSGI

gevent: Incomplete

class Thread(gevent.Greenlet):
    def __init__(self, target, args=[], kwargs={}) -> None: ...


class WebSocketWSGI:
    app: Incomplete
    def __init__(self, handler, server) -> None: ...
    environ: Incomplete
    version: Incomplete
    path: Incomplete
    origin: Incomplete
    protocol: Incomplete
    def __call__(self, environ, start_response): ...
    def close(self): ...
    def send(self, message): ...
    def wait(self): ...
