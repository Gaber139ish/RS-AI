import threading
import queue
from typing import Dict, List, Callable


class InProcBus:
    def __init__(self):
        self.queues: Dict[str, queue.Queue] = {}
        self.lock = threading.Lock()

    def subscribe(self, topic: str) -> queue.Queue:
        with self.lock:
            q = self.queues.get(topic)
            if q is None:
                q = queue.Queue(maxsize=1024)
                self.queues[topic] = q
            return q

    def publish(self, topic: str, msg):
        with self.lock:
            q = self.queues.get(topic)
            if q is not None:
                try:
                    q.put_nowait(msg)
                except queue.Full:
                    pass