from _typeshed import Incomplete

class Thread(Incomplete):
    def __init__(self, target, args=[], kwargs={}) -> None: ...

class uWSGIWebSocket:
    app: Incomplete
    received_messages: Incomplete
    def __init__(self, handler, server) -> None: ...
    environ: Incomplete
    def __call__(self, environ, start_response): ...
    def close(self) -> None: ...
    def send(self, msg) -> None: ...
    def wait(self): ...
