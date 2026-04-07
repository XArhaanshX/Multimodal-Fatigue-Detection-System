import asyncio
import json
import websockets
import time
import random

async def simulate_godot():
    """
    Simulates the Godot driving simulator by sending telemetry data over WebSockets
    and receiving reactive fatigue scores.
    """
    uri = "ws://localhost:8001"
    print(f"Connecting to IRoad Telemetry Server at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Successfully connected to simulator backend.")
            
            while True:
                # Generate physiologically plausible telemetry metrics
                # simulating slightly unsafe driving behavior
                telemetry = {
                    "lane_drift_var": round(random.uniform(0.1, 0.4), 3),
                    "lane_offset_mean": round(random.uniform(0.2, 0.5), 3),
                    "steering_instability": round(random.uniform(0.15, 0.35), 3),
                    "correction_freq": round(random.uniform(1.0, 4.0), 1),
                    "reaction_delay_mean": round(random.uniform(0.3, 0.7), 3),
                    "steering_reversals": random.randint(2, 8)
                }
                
                print(f"\n[CLIENT] Sending Telemetry: {telemetry}")
                await websocket.send(json.dumps(telemetry))
                
                # Wait for the reactive response from the server
                # The response is sent immediately after the update
                response_json = await websocket.recv()
                response = json.loads(response_json)
                
                print(f"[SERVER] Response received: Fatigue={response['fatigue_score']} | Alert={response['alert']}")
                
                if response['alert']:
                    print(">>> !!! DRIVER ALERT TRIGGERED IN SIMULATOR !!! <<<")
                
                await asyncio.sleep(1.0) # Send updates at 1 Hz
                
    except ConnectionRefusedError:
        print(f"[ERROR] Connection refused. Is run_fatigue_pipeline.py running?")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(simulate_godot())
