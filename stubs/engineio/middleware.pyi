from _typeshed import Incomplete
from engineio.static_files import get_static_file as get_static_file

class WSGIApp:
    engineio_app: Incomplete
    wsgi_app: Incomplete
    engineio_path: Incomplete
    static_files: Incomplete
    def __init__(self, engineio_app, wsgi_app: Incomplete | None = None, static_files: Incomplete | None = None, engineio_path: str = 'engine.io') -> None: ...
    socket: Incomplete
    def __call__(self, environ, start_response): ...
    def not_found(self, start_response): ...

class Middleware(WSGIApp):
    def __init__(self, engineio_app, wsgi_app: Incomplete | None = None, engineio_path: str = 'engine.io') -> None: ...
