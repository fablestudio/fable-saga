from _typeshed import Incomplete

class BaseSocket:
    upgrade_protocols: Incomplete
    server: Incomplete
    sid: Incomplete
    queue: Incomplete
    last_ping: Incomplete
    connected: bool
    upgrading: bool
    upgraded: bool
    closing: bool
    closed: bool
    session: Incomplete
    def __init__(self, server, sid) -> None: ...
