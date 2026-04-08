import asyncio
import websockets
import json

async def listen_to_fatigue():
    uri = "ws://localhost:8000/ws/fatigue-score"
    print(f"Connecting to {uri}...")
    try:
        async with websockets.connect(uri) as websocket:
            print("Connected! Waiting for fatigue scores...")
            while True:
                message = await websocket.recv()
                data = json.loads(message)
                score = data.get("fatigue_score")
                state = data.get("fatigue_state")
                print(f"[LIVE] Score: {score:.4f} | State: {state}")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: Make sure the server (main.py) is running first!")

if __name__ == "__main__":
    try:
        asyncio.run(listen_to_fatigue())
    except KeyboardInterrupt:
        print("\nStopped listener.")
