import json
import random
import string
import threading
import time
from queue import Queue, Empty

from flask import Flask, request, jsonify, Response
from typing import Dict, Callable

from inference.sampler import AutoregressiveSampler
from models.moduleapi import ILanguageModel
from tokenization.tokenizer import Tokenizer

app = Flask(__name__)


class InvalidRequestError(Exception):
    def __init__(self, message, param=None, code=None):
        self.message = message
        self.param = param
        self.code = code

    def to_json(self):
        return {
            "error": {
                "message": self.message,
                "type": "invalid_request_error",
                "param": self.param,
                "code": self.code
            }
        }


class ServedModel:
    def __init__(self, model: ILanguageModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = AutoregressiveSampler(model, tokenizer)

    def generate_text(self, prompt: str, num_tokens: int, temperature: float = 1.0, top_k: int = 0) -> str:
        return self.sampler.generate_text(prompt, num_tokens, temperature, top_k)

    def generate_text_stream(self, prompt: str, num_tokens: int, text_callback: Callable[[str], None],
                             temperature: float = 1.0, top_k: int = 0):
        return self.sampler.stream_text(prompt, num_tokens, text_callback, temperature, top_k)


# Suppose we have model instances and tokenizer instances for all models
models: Dict[str, ServedModel] = {}


@app.route('/api/v1/models', methods=['GET'])
def list_models():
    if not models:
        return jsonify({"data": [], "object": "list"})

    response_data = {
        "data": [
            {
                "id": model_id,
                "object": "model",
                "owned_by": "organization-owner",
                "permission": []
            } for model_id in models.keys()
        ],
        "object": "list"
    }

    return jsonify(response_data)


@app.route('/api/v1/completions', methods=['POST'])
def create_completion():
    try:
        data = request.get_json()

        if not data:
            raise InvalidRequestError("Invalid request body")

        model_id = data.get('model')
        prompt = data.get('prompt')
        max_tokens = data.get('max_tokens')
        temperature = data.get('temperature')
        stream = data.get('stream', False)

        validate(model_id, str, 'model')
        validate(prompt, str, 'prompt')
        validate(max_tokens, int, 'max_tokens')
        validate(temperature, float, 'temperature')

        if model_id not in models:
            raise InvalidRequestError(f"The model `{model_id}` does not exist")

        model = models[model_id]

        def generate_compl_id():
            characters = string.ascii_letters + string.digits
            random_string = 'cmpl-' + ''.join(random.choice(characters) for _ in range(24))
            return random_string

        completion_id = generate_compl_id()

        num_tokens = max_tokens - model.tokenizer.get_num_tokens(prompt)

        if stream:
            queue = Queue()
            generation_finished = False

            def launch_text_generation():
                nonlocal generation_finished
                model.generate_text_stream(prompt, num_tokens,
                                           lambda new_text: queue.put_nowait(new_text),
                                           temperature)
                generation_finished = True

            def generate_text_stream():
                threading.Thread(
                    target=launch_text_generation,
                    name="TextGeneration-" + completion_id,
                    daemon=True
                ).start()

                while not generation_finished:
                    try:
                        text = queue.get(block=True, timeout=1)
                    except Empty:
                        continue
                    completion_json = {
                        "id": completion_id,
                        "object": "text_completion",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [
                            {
                                "text": f"{text}",
                                "index": 0,
                                "logprobs": None,
                                "finish_reason": None
                            }
                        ]
                    }
                    yield f"data: {json.dumps(completion_json)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(generate_text_stream(), content_type="text/event-stream")

        text_completion = model.generate_text(prompt, max_tokens, temperature)

        response_data = {
            "id": completion_id,
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "text": f"{text_completion}",
                    "index": 0,
                    "logprobs": None,
                    "finish_reason": "length"
                }
            ],
            "usage": {
                "prompt_tokens": model.tokenizer.get_num_tokens(prompt),
                "completion_tokens": model.tokenizer.get_num_tokens(text_completion),
                "total_tokens": model.tokenizer.get_num_tokens(prompt + text_completion)
            }
        }

        return jsonify(response_data)
    except InvalidRequestError as e:
        return jsonify(e.to_json()), 400


def validate(value, value_type, param):
    if value is None or not isinstance(value, value_type):
        raise InvalidRequestError(f"'{value}' is not of type '{value_type.__name__}' - '{param}'")


if __name__ == '__main__':
    app.run()
