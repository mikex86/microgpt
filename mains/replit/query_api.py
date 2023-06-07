import openai

openai.api_base = "http://localhost:5000/api/v1"
openai.api_key = "sk-1234"

if __name__ == '__main__':
    response = openai.ChatCompletion.create(
        model="replit-3b",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Can you write a basic GPT2 implementation in pytorch?"},
        ],
        stream=True
    )
    for chunk in response:
        chunk_message = chunk['choices'][0]['delta']
        if 'content' in chunk_message:
            content = chunk_message['content']
            print(content, end='', flush=True)

    # response = openai.Completion.create(
    #     model='replit-3b',
    #     prompt='class Gpt2(torch.nn.Module):',
    #     max_tokens=1024,
    #     temperature=0.9,
    #     stream=True
    # )
    # generation_times = []
    # last_now = None
    # for event in response:
    #     now = event["created"] # type: ignore
    #     if last_now is not None:
    #         generation_times.append(now - last_now)
    #     last_now = now
    #     event_text = event['choices'][0]['text']  # type: ignore
    #     print(event_text, end='', flush=True)
    #
    # print()
    # print(f"Avg. tokens/s: {1 / (sum(generation_times) / len(generation_times))}")
