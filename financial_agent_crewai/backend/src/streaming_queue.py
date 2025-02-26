from io import StringIO
from queue import Queue


class StreamToQueue:
    """Redirects stdout to a queue to capture logs in real-time."""

    def __init__(self, queue: Queue) -> None:
        self.queue = queue
        self._buffer = StringIO()

    def write(self, message: str) -> None:
        self.queue.put(message)

    def flush(self) -> None:
        pass