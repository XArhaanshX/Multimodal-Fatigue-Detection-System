import asyncio
import json
import websockets
import threading
import time

# Centralized shared state for telemetry and fatigue results
shared_state = {
    "telemetry": {
        "lane_drift_var": 0.0,
        "lane_offset_mean": 0.0,
        "steering_instability": 0.0,
        "correction_freq": 0.0,
        "reaction_delay_mean": 0.0,
        "steering_reversals": 0.0
    },
    "latest_fatigue_score": 0.0,
    "alert": False,
    "last_telemetry_time": 0.0  # Monitoring connection health
}

# Thread lock for safe telemetry access between the WebSocket and ML threads
telemetry_lock = threading.Lock()

async def telemetry_handler(websocket, path=None):
    """
    Handles WebSocket connections from the Godot simulator.
    Uses thread-safe locking and provides reactive fatigue results.
    """
    remote_addr = websocket.remote_address
    print(f"[INFO] Simulator client connected from {remote_addr}")
    
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                
                # Thread-safe telemetry update (Change 1)
                with telemetry_lock:
                    for key in shared_state["telemetry"]:
                        if key in data:
                            shared_state["telemetry"][key] = float(data[key])
                    
                    shared_state["last_telemetry_time"] = time.time()
                
                # Print debug only on update to avoid spamming
                # print(f"[DEBUG] Telemetry Received from {remote_addr}")
                
                # Reactive Response: Send the latest fatigue score and alert status back to Godot
                response = {
                    "fatigue_score": round(float(shared_state["latest_fatigue_score"]), 4),
                    "alert": bool(shared_state["alert"])
                }
                await websocket.send(json.dumps(response))
                
            except json.JSONDecodeError:
                print(f"[ERROR] Received invalid JSON from {remote_addr}")
            except Exception as e:
                print(f"[ERROR] Error processing simulator message: {e}")
                
    except websockets.exceptions.ConnectionClosed:
        print(f"[INFO] Simulator client disconnected: {remote_addr}")
    except Exception as e:
        print(f"[ERROR] WebSocket handler error: {e}")

async def main():
    """
    Starts the WebSocket server on port 8001.
    """
    async with websockets.serve(telemetry_handler, "0.0.0.0", 8001):
        print("[INFO] Telemetry WebSocket Server listening on ws://0.0.0.0:8001")
        await asyncio.Future()  # run forever

def run_server():
    """
    Entry point for the background thread to launch the async server.
    """
    asyncio.run(main())

if __name__ == "__main__":
    run_server()
