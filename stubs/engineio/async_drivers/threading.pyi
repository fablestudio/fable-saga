import threading
from engineio.async_drivers._websocket_wsgi import SimpleWebSocketWSGI as SimpleWebSocketWSGI

class DaemonThread(threading.Thread):
    def __init__(self, *args, **kwargs) -> None: ...
