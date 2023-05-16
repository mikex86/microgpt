from typing import List

from environments.shell.terminal_provider import TerminalProvider


def capture_terminal_context(terminal_provider: TerminalProvider) -> bytes:
    term_context: List[str] = terminal_provider.get_terminal_context().copy()

    # Use ISO-8859-1 encoding to convert to bytes
    byte_rows: List[List[int]] = [list(row.encode('ISO-8859-1', 'replace')) for row in term_context]

    # visualize cursor position
    # TODO: RIP non-ASCII characters, however the convenience of constant
    #  length context prevails for now
    x, y = terminal_provider.get_terminal_cursor_position()

    # Place enquiry character at cursor position
    # I don't know if this is a historically correct way to use this character,
    # but it will work for our purposes
    byte_rows[y][x] = 5

    context_str_bytes = b''
    for row in byte_rows:
        context_str_bytes += bytes(row) + b'\n'

    return context_str_bytes