import threading
from typing import Callable, Dict

import zmq


class MessageBus:
    """Simple PUB/SUB message bus using ZeroMQ."""

    def __init__(self, address: str = "tcp://127.0.0.1:5555") -> None:
        self.ctx = zmq.Context.instance()
        self.address = address
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.bind(address)
        self._subs: Dict[str, zmq.Socket] = {}
        self._threads: list[threading.Thread] = []

    def publish(self, topic: str, data: bytes) -> None:
        """Broadcast ``data`` under ``topic``."""
        self.pub.send_multipart([topic.encode(), data])

    def subscribe(self, topic: str, handler: Callable[[bytes], None]) -> None:
        """Listen on ``topic`` and invoke ``handler`` for each message."""
        sub = self.ctx.socket(zmq.SUB)
        sub.connect(self.address)
        sub.setsockopt_string(zmq.SUBSCRIBE, topic)

        def loop() -> None:
            while True:
                _t, msg = sub.recv_multipart()
                handler(msg)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        self._subs[topic] = sub
        self._threads.append(t)
