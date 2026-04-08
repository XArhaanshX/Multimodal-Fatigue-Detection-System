import asyncio
import json
import websockets
import time
import random

async def simulate_godot():
    """
    Simulates the Godot driving simulator by sending high-frequency telemetry
    and receiving rate-limited fatigue updates.
    """
    uri = "ws://localhost:8001/ws/telemetry"
    print(f"Connecting to IRoad Telemetry Server at {uri}...")
    
    try:
        async with websockets.connect(uri) as websocket:
            print("Successfully connected to simulator backend.")
            
            async def send_telemetry():
                print("[INFO] Starting telemetry stream (10 Hz)...")
                while True:
                    # Randomly simulate either safe or fatigued behavior
                    mode = random.choice(["safe", "safe", "dangerous"]) 
                    
                    if mode == "safe":
                        telemetry = {
                            "lane_drift_var": round(random.uniform(0.01, 0.05), 3),
                            "lane_offset_mean": round(random.uniform(0.0, 0.1), 3),
                            "steering_instability": round(random.uniform(0.01, 0.05), 3),
                            "correction_freq": round(random.uniform(0.5, 1.5), 1),
                            "reaction_delay_mean": round(random.uniform(0.2, 0.4), 3),
                            "steering_reversals": random.randint(1, 4),
                            "steering_angle": round(random.uniform(-0.1, 0.1), 3),
                            "lane_offset": round(random.uniform(-0.2, 0.2), 3)
                        }
                    else:
                        telemetry = {
                            "lane_drift_var": round(random.uniform(0.4, 0.8), 3),
                            "lane_offset_mean": round(random.uniform(0.5, 1.2), 3),
                            "steering_instability": round(random.uniform(0.4, 0.9), 3),
                            "correction_freq": round(random.uniform(3.0, 6.0), 1),
                            "reaction_delay_mean": round(random.uniform(0.8, 1.2), 3),
                            "steering_reversals": random.randint(8, 15),
                            "steering_angle": round(random.uniform(-0.5, 0.5), 3),
                            "lane_offset": round(random.uniform(-1.0, 1.0), 3)
                        }
                    
                    await websocket.send(json.dumps(telemetry))
                    await asyncio.sleep(0.1) # 10 Hz

            async def receive_updates():
                print("[INFO] Listening for fatigue updates...")
                last_msg_time = time.time()
                while True:
                    try:
                        response_json = await websocket.recv()
                        data = json.loads(response_json)
                        
                        now = time.time()
                        interval = now - last_msg_time
                        last_msg_time = now
                        
                        if data.get("type") == "fatigue_update":
                            score = data["fatigue_score"]
                            state = data["fatigue_state"]
                            print(f"\n[SERVER] Update (@{interval:.2f}s): Score={score:.4f} | State={state.upper()}")
                            
                            if state in ["severe", "critical"]:
                                print(">>> !!! DRIVER ALERT: HAPTIC FEEDBACK TRIGGERED !!! <<<")
                        elif data.get("type") == "error":
                            print(f"\n[SERVER ERROR] {data['message']}")
                        else:
                            print(f"\n [SERVER] Unknown message: {data}")
                            
                    except Exception as e:
                        print(f"[ERROR] Receiver error: {e}")
                        break

            # Run both tasks concurrently
            await asyncio.gather(send_telemetry(), receive_updates())
                
    except ConnectionRefusedError:
        print(f"[ERROR] Connection refused. Is backend/network/backend.py running?")
    except Exception as e:
        print(f"[ERROR] Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(simulate_godot())
