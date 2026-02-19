from flask import Flask, Response
import time

app = Flask(__name__)

mock_events = [
    'data: {"choices":[{"delta":{"content":"Tesla","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1771518114,"id":"283ffbf5-591e-4187-9361-391602be5586","model":"Meta-Llama-3.1-8B-Instruct","object":"chat.completion.chunk","system_fingerprint":"fastcoe"}',
    'data: {"choices":[{"delta":{"content":", Inc. is an American multinational corporation that specializes in electric vehicles, clean energy generation and storage products","role":"assistant"},"finish_reason":null,"index":0,"logprobs":null}],"created":1771518114,"id":"283ffbf5-591e-4187-9361-391602be5586","model":"Meta-Llama-3.1-8B-Instruct","object":"chat.completion.chunk","system_fingerprint":"fastcoe"}',
    'data: {"choices":[{"delta":{"content":""},"finish_reason":"stop","index":0,"logprobs":null}],"created":1771518115,"id":"283ffbf5-591e-4187-9361-391602be5586","model":"Meta-Llama-3.1-8B-Instruct","object":"chat.completion.chunk","system_fingerprint":"fastcoe"}',
    'data: {"choices":[],"created":1771518115,"id":"283ffbf5-591e-4187-9361-391602be5586","model":"Meta-Llama-3.1-8B-Instruct","object":"chat.completion.chunk","system_fingerprint":"fastcoe","usage":{"prompt_tokens": 41,"completion_tokens": 108,"total_tokens": 149,"prompt_tokens_details": null},"fhenom_usage":{"total_encryption_time_ms":0.6704330444335938,"total_decryption_time_ms":0.3361701965332031,"server_network_latency_ms":1295.6573963165283}}',
    'data: [DONE]'
]

@app.route("/stream", methods=["POST"])
def stream():
    def event_stream():
        for event in mock_events:
            yield event + "\n\n"   # SSE format requires double newline
            time.sleep(0.1)
    return Response(event_stream(), mimetype="text/event-stream")

if __name__ == "__main__":
    app.run(port=8000)