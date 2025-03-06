from io import StringIO
from queue import Queue


class StreamToQueue:
    """
    A class that redirects stdout to a queue for real-time log capturing.

    This class is useful for redirecting printed output (typically to stdout)
    to a queue, allowing other parts of the program to consume log messages
    as they are generated, without blocking the main thread or writing to the console.

    Attributes:
      queue (Queue): The queue where log messages are placed.
      _buffer (StringIO): A buffer to handle incoming message data.
    """

    def __init__(self, queue: Queue[str]) -> None:
        """
        Initializes the StreamToQueue object with a given queue.

        Args:
          queue: The queue where log messages will be stored.
        """
        self.queue = queue
        self._buffer = StringIO()

    def write(self, message: str) -> None:
        """
        Writes a message to the queue.

        Args:
          message: The message to be added to the queue.
        """
        self.queue.put(message)

    def flush(self) -> None:
        pass
