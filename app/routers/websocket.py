from fastapi import APIRouter, WebSocket, WebSocketDisconnect
import json

router = APIRouter()

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_json()
            if data.get("type") == "ping":
                await websocket.send_json({"type": "pong"})
            else:
                await websocket.send_json({"type": "unknown", "data": data})
    except WebSocketDisconnect:
        pass 