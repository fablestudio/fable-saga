from . import packet as packet, payload as payload
from _typeshed import Incomplete

default_logger: Incomplete

class BaseServer:
    compression_methods: Incomplete
    event_names: Incomplete
    valid_transports: Incomplete
    sequence_number: int
    ping_timeout: Incomplete
    ping_interval: Incomplete
    ping_interval_grace_period: Incomplete
    max_http_buffer_size: Incomplete
    allow_upgrades: Incomplete
    http_compression: Incomplete
    compression_threshold: Incomplete
    cookie: Incomplete
    cors_allowed_origins: Incomplete
    cors_credentials: Incomplete
    async_handlers: Incomplete
    sockets: Incomplete
    handlers: Incomplete
    log_message_keys: Incomplete
    start_service_task: Incomplete
    service_task_handle: Incomplete
    service_task_event: Incomplete
    logger: Incomplete
    async_mode: Incomplete
    transports: Incomplete
    def __init__(self, async_mode: Incomplete | None = None, ping_interval: int = 25, ping_timeout: int = 20, max_http_buffer_size: int = 1000000, allow_upgrades: bool = True, http_compression: bool = True, compression_threshold: int = 1024, cookie: Incomplete | None = None, cors_allowed_origins: Incomplete | None = None, cors_credentials: bool = True, logger: bool = False, json: Incomplete | None = None, async_handlers: bool = True, monitor_clients: Incomplete | None = None, transports: Incomplete | None = None, **kwargs) -> None: ...
    def is_asyncio_based(self): ...
    def async_modes(self): ...
    def on(self, event, handler: Incomplete | None = None): ...
    def transport(self, sid): ...
    def create_queue(self, *args, **kwargs): ...
    def get_queue_empty_exception(self): ...
    def create_event(self, *args, **kwargs): ...
    def generate_id(self): ...
