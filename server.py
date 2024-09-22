import asyncio
import re
import websockets
from flask import Flask, Response, request
from flask_cors import CORS
from flask_sock import Sock
import requests
from simple_websocket import Server
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
import pickledb

db = pickledb.load('messages.db', True)

app = Flask(__name__)
sock = Sock(app)

CORS(app)

model_name = "croissantllm/CroissantLLMChat-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)


generation_args = {
    "max_new_tokens": 256,
    "do_sample": True,
    "temperature": 0.3,
    "top_p": 0.90,
    "top_k": 40,
    "repetition_penalty": 1.05,
    "eos_token_id": [tokenizer.eos_token_id, 32000],
}


def run_translate_prompt(prompt: str, text: str) -> str:
    chat = [
        {
            "role": "user",
            "content": f"{prompt}: \"{text}\"",
        },
    ]

    chat_input = tokenizer.apply_chat_template(
        chat, tokenize=False, add_generation_prompt=True
    )

    inputs = tokenizer(chat_input, return_tensors="pt").to(model.device)
    tokens = model.generate(**inputs, **generation_args)

    raw_text = tokenizer.decode(tokens[0])

    # extract the assistant's text from this
    output = raw_text.split('assistant\n')[-1].split("<|im_end|>")[0].strip().strip("\"")
    response = re.sub(r'^[\'"“”‘’«\s]+|[\'"“”»‘’\s.,!?]+$', '', output).strip()
    return response


def french_to_english(text: str) -> str:
    return run_translate_prompt("Translate this French to English", text)


def english_to_french(text: str) -> str:
    return run_translate_prompt("Translate this English to French", text)

# Target API base URL
TARGET_API_DOMAIN = "qcs.shsbs.xyz"


def forward_request_to_target():
    # Construct the target URL
    target_url = "https://" + TARGET_API_DOMAIN + request.path

    # Capture all headers, data, and method from the incoming request
    headers = {key: value for key, value in request.headers if key.lower() != "host"}
    data = request.get_data()

    try:
        # Forward the request to the target API
        response = requests.request(
            method=request.method,
            url=target_url,
            headers=headers,
            data=data,
            params=request.args,
            stream=True,  # This allows us to stream the response
        )

        # Create a Flask Response object
        excluded_headers = [
            "content-encoding",
            "content-length",
            "transfer-encoding",
            "connection",
        ]
        headers = [
            (name, value)
            for (name, value) in response.raw.headers.items()
            if name.lower() not in excluded_headers
        ]

        return Response(
            response.iter_content(chunk_size=10 * 1024),
            status=response.status_code,
            headers=headers,
        )

    except requests.RequestException as e:
        # Handle any requests exceptions
        return Response(f"Error forwarding request: {str(e)}", status=500)


async def client_to_server(ws: Server, websocket: websockets.WebSocketClientProtocol):
    """Forward messages from the client to the server."""

    try:
        while True:
            # Use asyncio to handle sending messages
            print("CLIENT", "WAITING")
            client_message = await asyncio.to_thread(ws.receive)
            if client_message is None:
                break
            print("CLIENT", client_message)
            await websocket.send(client_message)
    except websockets.ConnectionClosed:
        print("CLIENT", "WebSocket connection closed")
    except Exception as e:
        print("CLIENT", f"Exception: {e}")


async def server_to_client(ws: Server, websocket: websockets.WebSocketClientProtocol):
    """Forward messages from the server to the client."""
    try:
        while True:
            print("SERVER", "WAITING")
            server_message = await websocket.recv()
            print("SERVER", server_message)
            data = json.loads(server_message)
            msg_type = data['type']
            try:
                if msg_type == 'message_event' or msg_type == 'live':
                    for index, message in enumerate(data['data']['objects']['message_event']['message']):
                        key = f"msg-{message['id']}"
                        if db.exists(key):
                            message['text'] = db.get(key)
                        else:
                            message['text'] = english_to_french(message['text'])
                            db.set(key, message['text'])
                        data['data']['objects']['message_event']['message'][index] = message
                elif msg_type == 'request' and 'message' in data['data']['objects']:
                    for index, message in enumerate(data['data']['objects']['message']):
                        key = f"msg-{message['id']}"
                        if db.exists(key):
                            message['text'] = db.get(key)
                        else:
                            message['text'] = english_to_french(message['text'])
                            db.set(key, message['text'])
                        data['data']['objects']['message'][index] = message
            except KeyError as e:
                print(e)
            ws.send(json.dumps(data))
    except websockets.ConnectionClosed:
        pass  # Handle server disconnection


@sock.route("/api/live/ws")
def ws_handle(ws):
    query_string = request.query_string.decode("utf-8")
    ws_url = f"wss://{TARGET_API_DOMAIN}/api/live/ws?{query_string}"

    async def handle_ws():
        async with websockets.connect(ws_url) as websocket:
            # Run both client-to-server and server-to-client concurrently
            await asyncio.gather(
                server_to_client(ws, websocket),
                client_to_server(ws, websocket),
            )

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(handle_ws())


# Define routes for all HTTP methods
@app.route(
    "/<path:path>", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "OPTIONS", "HEAD"]
)
def catch_all(path):
    return forward_request_to_target()


if __name__ == "__main__":
    app.run(debug=True, port=5000)
