from _typeshed import Incomplete

OPEN: Incomplete
CLOSE: Incomplete
PING: Incomplete
PONG: Incomplete
MESSAGE: Incomplete
UPGRADE: Incomplete
NOOP: Incomplete
packet_names: Incomplete
binary_types: Incomplete

class Packet:
    json: Incomplete
    packet_type: Incomplete
    data: Incomplete
    encode_cache: Incomplete
    binary: bool
    def __init__(self, packet_type=..., data: Incomplete | None = None, encoded_packet: Incomplete | None = None) -> None: ...
    def encode(self, b64: bool = False): ...
    def decode(self, encoded_packet) -> None: ...
