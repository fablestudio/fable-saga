from . import packet as packet
from _typeshed import Incomplete

default_logger: Incomplete
connected_clients: Incomplete

def signal_handler(sig, frame): ...

original_signal_handler: Incomplete

class BaseClient:
    event_names: Incomplete
    handlers: Incomplete
    base_url: Incomplete
    transports: Incomplete
    current_transport: Incomplete
    sid: Incomplete
    upgrades: Incomplete
    ping_interval: Incomplete
    ping_timeout: Incomplete
    http: Incomplete
    external_http: Incomplete
    handle_sigint: Incomplete
    ws: Incomplete
    read_loop_task: Incomplete
    write_loop_task: Incomplete
    queue: Incomplete
    state: str
    ssl_verify: Incomplete
    websocket_extra_options: Incomplete
    logger: Incomplete
    request_timeout: Incomplete
    def __init__(self, logger: bool = False, json: Incomplete | None = None, request_timeout: int = 5, http_session: Incomplete | None = None, ssl_verify: bool = True, handle_sigint: bool = True, websocket_extra_options: Incomplete | None = None) -> None: ...
    def is_asyncio_based(self): ...
    def on(self, event, handler: Incomplete | None = None): ...
    def transport(self): ...
