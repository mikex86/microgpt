from asyncio import Queue

from environments.shell.dataset.terminal_capture import capture_terminal_context
from environments.shell.terminal_provider import TerminalProvider
import struct


class TerminalDatasetRecorder:

    def __init__(self, dataset_file_path: str, terminal_provider: TerminalProvider):
        self.dataset_file = open(dataset_file_path, 'wb+')
        self.terminal_provider = terminal_provider
        self.prev_terminal_context = None

        # A queue, where context byte strings are queue up for "clearance".
        # A context has been "cleared", when no handle_input() calls have been made
        # while it was the current state, meaning the terminal context required no input action from the user.
        self.context_clearance_queue: Queue[bytes] = Queue()
        self._write_header()

    def _write_header(self):
        # write terminal size as int32 big endian
        width, height = self.terminal_provider.get_terminal_size()
        self.dataset_file.write(struct.pack('>i', width))
        self.dataset_file.write(struct.pack('>i', height))
        self.dataset_file.flush()

    def handle_input(self, new_stdin: str):
        context_str_bytes = capture_terminal_context(self.terminal_provider)

        buffer = []
        # clear the context_clearance_queue
        while not self.context_clearance_queue.empty():
            context_bytes = self.context_clearance_queue.get_nowait()
            if context_bytes == context_str_bytes:
                break
            buffer.append(context_bytes)

        for context_bytes in buffer:
            self.__make_no_action_entry(context_bytes)

        width, height = self.terminal_provider.get_terminal_size()

        if len(context_str_bytes) != ((width + 1) * height):  # +1 for newline
            raise Exception('Terminal context string in bytes does not match terminal size product')

        new_stdin_bytes = new_stdin.encode('ISO-8859-1', 'replace')

        if len(new_stdin_bytes) != 1:
            # TODO:
            # in this format, we only support single character inputs
            # this should be as a given when the source of input is a TerminalGui() object and
            # you don't paste.
            raise Exception('New stdin string in bytes is not length 1')

        self.__make_entry(new_stdin_bytes, context_str_bytes)
        self.dataset_file.flush()

    def update(self):
        context_str_bytes = capture_terminal_context(self.terminal_provider)
        if context_str_bytes == self.prev_terminal_context:
            return
        self.context_clearance_queue.put_nowait(context_str_bytes)
        self.prev_terminal_context = context_str_bytes

    def __make_no_action_entry(self, context_str_bytes: bytes):
        new_stdin_bytes = b'\x00'

        self.__make_entry(new_stdin_bytes, context_str_bytes)

    def __make_entry(self, new_stdin_bytes: bytes, context_str_bytes: bytes):
        self.dataset_file.write(new_stdin_bytes)
        self.dataset_file.write(context_str_bytes)
        self.dataset_file.flush()

    def close(self):
        self.dataset_file.close()
