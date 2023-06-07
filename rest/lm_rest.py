import json
import random
import string
import threading
import time
from queue import Queue, Empty

from flask import Flask, request, jsonify, Response
from typing import Dict, Callable, List, Union, Optional

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
    def __init__(self, model: ILanguageModel, tokenizer: Tokenizer, max_seq_len: int):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = AutoregressiveSampler(model, tokenizer)
        self.max_seq_len = max_seq_len

    def generate_text(self, prompt: str, num_tokens: int, temperature: float = 1.0, top_k: int = 0,
                      stop: Optional[Union[List[str], str]] = None) -> str:
        return self.sampler.generate_text(prompt, num_tokens, temperature, top_k, stop=stop, include_prompt=False)

    def generate_text_stream(self, prompt: str, num_tokens: int, text_callback: Callable[[str], None],
                             temperature: float = 1.0, top_k: int = 0, stop: Optional[Union[List[str], str]] = None):
        if stop is None:
            stop = []
        return self.sampler.stream_text(prompt, num_tokens, text_callback, temperature, top_k, stop=stop)


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
        stop = data.get('stop', [])
        stream = data.get('stream', False)

        validate(model_id, str, 'model')
        validate(prompt, str, 'prompt')
        validate(max_tokens, int, 'max_tokens')
        validate(temperature, float, 'temperature')
        validate(stop, [list, str], 'stop')

        if model_id not in models:
            raise InvalidRequestError(f"The model `{model_id}` does not exist")

        model = models[model_id]

        def _generate_compl_id():
            characters = string.ascii_letters + string.digits
            random_string = 'cmpl-' + ''.join(random.choice(characters) for _ in range(24))
            return random_string

        completion_id = _generate_compl_id()

        num_tokens = max_tokens - model.tokenizer.get_num_tokens(prompt)
        if num_tokens < 0:
            raise InvalidRequestError("The prompt is too long for the given max_tokens")

        if stream:
            queue = Queue()
            generation_finished = False

            def launch_text_generation():
                nonlocal generation_finished
                model.generate_text_stream(prompt, num_tokens,
                                           lambda new_text: queue.put_nowait(new_text),
                                           temperature,
                                           stop=stop)
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


@app.route('/api/v1/chat/completions', methods=['POST'])
def create_chat_completion():
    try:
        data = request.get_json()

        if not data:
            raise InvalidRequestError("Invalid request body")

        model_id = data.get('model')
        messages = data.get('messages')
        temperature = data.get('temperature', 0.7)
        stream = data.get('stream', False)
        stop = data.get('stop', [])

        validate(model_id, str, 'model')
        validate(messages, list, 'messages')
        validate(temperature, float, 'temperature')
        validate(stop, [list, str], 'stop')

        if model_id not in models:
            raise InvalidRequestError(f"The model `{model_id}` does not exist")

        model = models[model_id]

        def _generate_chatcompl_id():
            characters = string.ascii_letters + string.digits
            random_string = 'chatcmpl-' + ''.join(random.choice(characters) for _ in range(24))
            return random_string

        completion_id = _generate_chatcompl_id()

        prompt = ""
        for message in messages:
            if 'role' not in message:
                raise InvalidRequestError("Invalid message: missing 'role' field")
            if 'content' not in message:
                raise InvalidRequestError("Invalid message: missing 'content' field")

            role = message['role']
            content = message['content']

            prompt += f"{role}:\n {content}\n"
        prompt += "assistant:\n"

        # add roles used in the prompt to the stop list
        for role in set([message['role'] for message in messages]):
            stop.append(role + ":")

        if 'assistant:' not in stop:
            stop.append('assistant:')

        if stream:
            queue = Queue()
            generation_finished = False

            def launch_text_generation():
                nonlocal generation_finished
                model.generate_text_stream(prompt, model.max_seq_len,
                                           lambda new_text: queue.put_nowait(new_text),
                                           temperature,
                                           stop=stop)
                generation_finished = True

            def generate_text_stream():
                threading.Thread(
                    target=launch_text_generation,
                    name="TextGeneration-" + completion_id,
                    daemon=True
                ).start()

                # start assistant message
                completion_json = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_id,
                    "choices": [
                        {
                            "delta": {
                                "role": "assistant"
                            },
                        }
                    ]
                }
                yield f"data: {json.dumps(completion_json)}\n\n"

                # send other tokens as deltas
                while not generation_finished:
                    try:
                        text = queue.get(block=True, timeout=1)
                    except Empty:
                        continue
                    completion_json = {
                        "id": completion_id,
                        "object": "chat.completion.chunk",
                        "created": int(time.time()),
                        "model": model_id,
                        "choices": [
                            {
                                "delta": {
                                    "content": f"{text}"
                                },
                            }
                        ]
                    }
                    yield f"data: {json.dumps(completion_json)}\n\n"
                yield "data: [DONE]\n\n"

            return Response(generate_text_stream(), content_type="text/event-stream")
        else:
            text_completion = model.generate_text(prompt, max_tokens, temperature, stop=stop)

            response_data = {
                "id": completion_id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": model_id,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": text_completion
                        }
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
    if value is None or not (
            any([isinstance(value, type_element) for type_element in value_type])
            if isinstance(value_type, list)
            else isinstance(value, value_type)
    ):
        raise InvalidRequestError(f"'{value}' is not of type '{value_type.__name__}' - '{param}'")


if __name__ == '__main__':
    app.run()
