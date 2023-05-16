from typing import List

from environments.shell.terminal_provider import TerminalProvider
import struct


class TerminalDatasetRecorder:

    def __init__(self, dataset_file_path: str, terminal_provider: TerminalProvider):
        self.dataset_file = open(dataset_file_path, 'wb+')
        self.terminal_provider = terminal_provider
        self._write_header()

    def _write_header(self):
        # write terminal size as int32 big endian
        width, height = self.terminal_provider.get_terminal_size()
        self.dataset_file.write(struct.pack('>i', width))
        self.dataset_file.write(struct.pack('>i', height))
        self.dataset_file.flush()

    def handle_input(self, new_stdin: str):
        context_str_bytes = capture_terminal_context(self.terminal_provider)

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

        self.dataset_file.write(context_str_bytes)
        self.dataset_file.write(new_stdin_bytes)
        self.dataset_file.flush()

    def close(self):
        self.dataset_file.close()
