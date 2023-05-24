import random
import string
import time

from flask import Flask, request, jsonify, Response
from typing import Dict

from inference.sampler import AutoregressiveSampler
from models.moduleapi import ILanguageModel
from tokenization.tokenizer import Tokenizer

app = Flask(__name__)


class ServedModel:
    def __init__(self, model: ILanguageModel, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = AutoregressiveSampler(model, tokenizer)

    def generate_text(self, prompt: str, num_tokens: int, temperature: float = 1.0, top_k: int = 0) -> str:
        return self.sampler.generate_text(prompt, num_tokens, temperature, top_k)

    def generate_text_stream(self, prompt: str, num_tokens: int, temperature: float = 1.0, top_k: int = 0):
        texts = []

        def handle_text(text: str):
            texts.append(text)

        self.sampler.stream_text(prompt, num_tokens, handle_text, temperature, top_k)

        return texts


# Suppose we have model instances and tokenizer instances for all models
models: Dict[str, ServedModel] = {}


@app.route('/v1/models', methods=['GET'])
def list_models():
    response_data = {"data": [], "object": "list"}

    for model_id in models.keys():
        response_data["data"].append({
            "id": model_id,
            "object": "model",
            "owned_by": "organization-owner",
            "permission": []
        })

    return jsonify(response_data)


@app.route('/v1/completions', methods=['POST'])
def create_completion():
    data = request.get_json()

    model_id = data['model']
    prompt = data['prompt']
    max_tokens = data['max_tokens']
    temperature = data['temperature']
    stream = data['stream'] if 'stream' in data else False

    if model_id not in models:
        return {"error": "Invalid model id"}, 400

    model = models[model_id]

    def generate_compl_id():
        characters = string.ascii_letters + string.digits
        random_string = 'cmpl-' + ''.join(random.choice(characters) for _ in range(24))
        return random_string

    if stream:
        def generate_text_stream():
            texts = model.generate_text_stream(prompt, max_tokens, temperature)
            for text in texts:
                yield f"data: {text}\n\n"
            yield "data: [DONE]\n\n"

        return Response(generate_text_stream(), content_type="text/event-stream")

    text_completion = model.generate_text(prompt, max_tokens, temperature)

    response_data = {
        "id": generate_compl_id(),
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
