from threading import Thread

import pyglet
import torch

from environments.shell.dataset.terminal_capture import capture_terminal_context
from environments.shell.docker_terminal_provider import DockerTerminalProvider
from environments.shell.terminal_gui import TerminalGui
from inference.sampler import AutoregressiveSampler
from models.termgpt import TerminalGptConfig, TerminalGptModel
from tokenization.greedy_tokenizer import GreedyTokenizer
from train import checkpointing

TERMINAL_WIDTH = 80
TERMINAL_HEIGHT = 20


def main():
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')

    term_provider = DockerTerminalProvider('test_ubuntu', (TERMINAL_WIDTH, TERMINAL_HEIGHT))
    term_gui = TerminalGui('Terminal', term_provider, True)

    gpt_config = TerminalGptConfig(
        block_size=(TERMINAL_WIDTH + 1) * TERMINAL_HEIGHT + 1,
        n_layers=6,
        n_heads=8,
        n_embd=64,
        device=device,
        dtype=torch.float32,
        vocab_size=256
    )

    model = TerminalGptModel(gpt_config).to(device)
    checkpointing.load_checkpoint(model, None, 'checkpoints/termgpt/dummyds', 'best')

    # ISO-8859-1 tokenizer
    tokenizer = GreedyTokenizer(stoi={bytes([i]).decode('ISO-8859-1'): i for i in range(256)})
    sampler = AutoregressiveSampler(model, tokenizer)

    completion_running = False

    max_frame_delay = 8
    frame_delay = max_frame_delay

    last_term_ctx = None

    def check_start_completion():
        nonlocal completion_running, frame_delay, last_term_ctx
        if completion_running:
            return

        current_term_ctx = capture_terminal_context(term_provider)
        if current_term_ctx == last_term_ctx:
            return

        # give it some time to update the terminal
        if frame_delay > 0:
            frame_delay -= 1
            return

        def sample_model():
            nonlocal completion_running, frame_delay, last_term_ctx
            term_ctx = capture_terminal_context(term_provider)
            prompt = term_ctx.decode('ISO-8859-1')
            last_term_ctx = term_ctx
            text = sampler.generate_text(prompt, num_tokens=1, temperature=0.0, include_prompt=False)
            if text != '':
                term_provider.send_input(text)
            else:
                frame_delay = max_frame_delay
            frame_delay = max_frame_delay / 2
            completion_running = False

        completion_thread = Thread(target=sample_model, daemon=True, name='CompletionThread')
        completion_running = True
        completion_thread.start()

    pyglet.clock.schedule(lambda dt: term_provider.update())

    pyglet.clock.schedule(lambda dt: check_start_completion())

    pyglet.app.run()


if __name__ == '__main__':
    main()
