import pyglet

from docker_terminal_provider import DockerTerminalProvider
from environments.shell.dataset.terminal_dataset_recorder import TerminalDatasetRecorder
from terminal_gui import TerminalGui

TERMINAL_WIDTH = 80
TERMINAL_HEIGHT = 20

if __name__ == '__main__':
    term_provider = DockerTerminalProvider('test_ubuntu', (TERMINAL_WIDTH, TERMINAL_HEIGHT))

    dataset_path = 'recording_dataset.bin'

    dataset_recorder = TerminalDatasetRecorder(dataset_path, term_provider)

    # Save and exit
    term_gui = TerminalGui('Terminal', term_provider, True)

    def handle_input(new_stdin: str):
        dataset_recorder.handle_input(new_stdin)


    term_gui.add_input_listener(lambda new_stdin: handle_input(new_stdin))

    pyglet.clock.schedule(lambda dt: term_provider.update())

    pyglet.app.run()

    dataset_recorder.close()
