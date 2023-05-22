from typing import Callable, List

import pyglet
from .terminal_provider import TerminalProvider

key_xterm_mapping = {
    pyglet.window.key.ESCAPE: '\x1b',
    pyglet.window.key.F1: '\x1bOP',
    pyglet.window.key.F2: '\x1bOQ',
    pyglet.window.key.F3: '\x1bOR',
    pyglet.window.key.F4: '\x1bOS',
    pyglet.window.key.F5: '\x1b[15~',
    pyglet.window.key.F6: '\x1b[17~',
    pyglet.window.key.F7: '\x1b[18~',
    pyglet.window.key.F8: '\x1b[19~',
    pyglet.window.key.F9: '\x1b[20~',
    pyglet.window.key.F10: '\x1b[21~',
    pyglet.window.key.F11: '\x1b[23~',
    pyglet.window.key.F12: '\x1b[24~',
    pyglet.window.key.TAB: '\t'
}

motion_xterm_mapping = {
    pyglet.window.key.MOTION_UP: '\x1b[A',
    pyglet.window.key.MOTION_DOWN: '\x1b[B',
    pyglet.window.key.MOTION_LEFT: '\x1b[D',
    pyglet.window.key.MOTION_RIGHT: '\x1b[C',
    pyglet.window.key.MOTION_BACKSPACE: '\x7f',
    pyglet.window.key.MOTION_DELETE: '\x1b[3~',
}

is_running = False


class TerminalGui:

    def __init__(self, title: str, terminal_provider: TerminalProvider, enable_input: bool):
        self.title = title
        self.term_prov = terminal_provider
        self.enable_input = enable_input
        self.input_listeners: List[Callable[[str], None]] = []
        terminal_width, terminal_height = self.term_prov.get_terminal_size()

        font_size = 16
        self.window = pyglet.window.Window(width=int(terminal_width * font_size * 0.722),
                                           height=int(terminal_height * font_size * 1.5), caption=self.title,
                                           vsync=True)

        self.labels = []
        for i in range(terminal_height):
            label = pyglet.text.Label(text='', font_name='Consolas', font_size=font_size, x=0,
                                      y=self.window.height - i * font_size * 1.5, anchor_x='left', anchor_y='top')
            self.labels.append(label)

        @self.window.event
        def on_draw():
            self.window.clear()
            cursor_x, cursor_y = self.term_prov.get_terminal_cursor_position()
            lines = self.term_prov.get_terminal_context()

            for i, line in enumerate(lines):
                self.labels[i].text = line
                self.labels[i].draw()

            # draw cursor

            cursor_y = self.window.height - (cursor_y + 1) * font_size * 1.5
            cursor_x = cursor_x * font_size * 0.722

            shape = pyglet.shapes.Rectangle(x=cursor_x, y=cursor_y, width=font_size * 0.75, height=font_size * 1.5,
                                            color=(255, 255, 255))
            shape.draw()

        @self.window.event
        def on_key_press(symbol, modifiers):
            if symbol in key_xterm_mapping:
                stdin = key_xterm_mapping[symbol]
                for listener in self.input_listeners:
                    listener(stdin)
                if self.enable_input:
                    self.term_prov.send_input(stdin)
            if symbol == pyglet.window.key.ESCAPE:
                return pyglet.event.EVENT_HANDLED

        @self.window.event
        def on_text(text):
            for listener in self.input_listeners:
                listener(text)
            if self.enable_input:
                self.term_prov.send_input(text)

        @self.window.event
        def on_text_motion(motion):
            if motion in motion_xterm_mapping:
                stdin = motion_xterm_mapping[motion]
                if self.enable_input:
                    self.term_prov.send_input(stdin)

                for listener in self.input_listeners:
                    listener(stdin)

    def add_input_listener(self, listener: Callable[[str], None]):
        self.input_listeners.append(listener)
