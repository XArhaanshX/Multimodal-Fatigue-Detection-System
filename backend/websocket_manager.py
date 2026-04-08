from fastapi import WebSocket
from typing import List
import json

class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        """Broadcasts a JSON message to all connected clients."""
        # Create a copy of the list to avoid issues with concurrent removal
        for connection in self.active_connections[:]:
            try:
                await connection.send_json(message)
            except Exception:
                # Silently handle disconnected clients that haven't been removed yet
                self.active_connections.remove(connection)

manager = ConnectionManager()
