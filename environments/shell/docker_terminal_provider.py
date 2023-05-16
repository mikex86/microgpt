import os
from socket import socket
from time import sleep

import docker
import pyte

if os.name == "nt":
    import win32file
    import win32pipe
    import pywintypes

from .terminal_provider import TerminalProvider
from docker.models.containers import Container


class DockerTerminalProvider(TerminalProvider):

    def __init__(self, instance_name: str, terminal_size: (int, int) = (80, 43)):
        self.client = docker.from_env()
        containers = self.client.containers.list(filters={'name': instance_name})

        # Create if container does not exist yet
        if len(containers) == 0:
            self.container: Container = self.client.containers.run('ubuntu_custom', 'sh',
                                                                   name=instance_name,
                                                                   detach=True, tty=True, stdin_open=True,
                                                                   stdout=True, stderr=True,
                                                                   hostname='811423f3c78d')
        else:
            self.container = containers[0]
            # if container is not running, start it
            if self.container.status != 'running':
                self.container.start()
        result = self.container.exec_run('bash',
                                         tty=True, stdin=True, stdout=True,
                                         stderr=True, demux=True,
                                         socket=True,
                                         workdir='/root')
        self.exec_result = result

        self.terminal_width, self.terminal_height = terminal_size
        self.tty_socket = result.output
        self.open = True

        # wait for first byte from tty_socket
        if os.name == "nt":
            self._first_byte = self.tty_socket.recv(1)
        else:
            self._first_byte = self.tty_socket._sock.recv(1)

        # Reflect terminal size in container
        self.send_input(f"stty rows {self.terminal_height} cols {self.terminal_width} && clear\r")
        if os.name == 'nt':
            self.tty_socket._timeout = win32pipe.NMPWAIT_NOWAIT

        self._terminal_context = [" " * self.terminal_width] * self.terminal_height

        # create terminal emulator
        self.pyte_screen = pyte.Screen(self.terminal_width, self.terminal_height)
        self.pyte_stream = pyte.Stream(self.pyte_screen)

    def _poll_stream(self):
        buffer = b''
        if os.name == 'nt':
            received = b''
            handle = self.tty_socket._handle
            # peak available
            try:
                _, n_avail, _ = win32pipe.PeekNamedPipe(handle, 0)
                if n_avail == 0:
                    return
                while n_avail > 0:
                    error, new_received = win32file.ReadFile(handle, n_avail, None)
                    received += new_received
                    _, n_avail, _ = win32pipe.PeekNamedPipe(handle, 0)
            except pywintypes.error as e:
                # 109 = ERROR_BROKEN_PIPE
                if e.winerror == 109:
                    self.open = False
                    return
                if received == b'':
                    return
        else:
            try:
                received = self.tty_socket._sock.recv(1024)
            except socket.timeout:
                sleep(0.2)
                return
            except BrokenPipeError:
                self.open = False
                return
        if not received:
            return

        # handle first tts byte
        if self._first_byte is not None:
            received = self._first_byte + received
            self._first_byte = None

        # handle multibyte utf-8 characters
        if received[-1] & 0b10000000:
            buffer += received
            return
        else:
            buffer = buffer + received

        # build up buffer until size 1024 is reached
        self._update_terminal_context(buffer)

    def _update_terminal_context(self, tty_data: bytes):
        """
        Updates terminal context with tty_data
        """
        self.pyte_stream.feed(tty_data.decode('ISO-8859-1', 'ignore'))
        self.pyte_screen.ensure_vbounds()
        self.pyte_screen.ensure_hbounds()

        for idx, line in enumerate(self.pyte_screen.display, 1):
            self._terminal_context[idx - 1] = line

    def update(self):
        self._poll_stream()

    def send_input(self, input_str: str):
        try:
            if os.name == "nt":
                self.tty_socket.send(input_str.encode('utf-8'))
            else:
                self.tty_socket._sock.send(input_str.encode('utf-8'))
        except (BrokenPipeError, pywintypes.error):
            self.open = False

    def is_open(self) -> bool:
        if self.exec_result.exit_code is not None:
            self.open = False
        return self.open

    def shut_down(self):
        self.container.stop()
        self.container.remove()

    def get_terminal_size(self) -> (int, int):
        return self.terminal_width, self.terminal_height

    def get_terminal_context(self) -> [str]:
        return self._terminal_context

    def get_terminal_cursor_position(self) -> (int, int):
        cursor = self.pyte_screen.cursor
        return cursor.x, cursor.y

    def set_terminal_cursor_position(self, x: int, y: int):
        # TODO
        # self.send_input(f"\033[{y + 1};{x + 1}H")
        pass

    def join(self):
        pass
