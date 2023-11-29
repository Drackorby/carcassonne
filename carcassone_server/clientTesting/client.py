import asyncio
import websockets
import json


async def hello():
    uri = "ws://127.0.0.1:8000/action/1/"
    async with websockets.connect(uri) as websocket:
        name = input("What's your name? ")

        await websocket.send(json.dumps(name))
        print(f">>> {name}")

        greeting = json.loads(await websocket.recv())
        print(f"<<< {greeting}")

if __name__ == "__main__":
    asyncio.run(hello())