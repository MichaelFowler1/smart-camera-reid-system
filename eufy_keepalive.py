"""
eufy_keepalive.py
-----------------
Tiny daemon that keeps your Eufy S330 (= Floodlight Cam 2 Pro, T8423) in
continuous RTSP streaming mode by talking to a local eufy-security-ws
server. Run this in its own terminal alongside reid_node.py.

Why this exists:
    The S330's firmware only serves RTSP during motion events by default,
    so your camera's RTSP URL returns 404 most of the time. eufy-security-ws
    exposes a P2P command that flips the camera's internal RTSP server to
    "on and stays on." We send that command, then listen for the
    "stream stopped" event so we can re-kick it if the camera drops it.

Setup:
    1. Install Node.js LTS (nodejs.org)
    2. PowerShell:  npm install -g eufy-security-ws
    3. Create config.json next to where you run it:
         { "username": "you@example.com",
           "password": "your_eufy_password",
           "country": "US" }
    4. Run it in one terminal:  eufy-security-ws --config config.json
       (first run may need 2FA / CAPTCHA in the eufy-security-ws prompt)
    5. Find your S330's serial number in the Eufy app:
         Camera -> Settings -> Device Info -> Serial Number (T8423...)
    6. Add this line to your .env:
         EUFY_SERIAL=T8423XXXXXXXXXXXX
    7. pip install websockets python-dotenv
    8. python eufy_keepalive.py   (leave it running)
    9. Now run reid_node.py — your CAMERA_URL should respond continuously.
"""

import asyncio
import json
import os
import sys
import uuid

from dotenv import load_dotenv
import websockets

load_dotenv()

SERIAL = os.getenv("EUFY_SERIAL")
WS_URL = os.getenv("EUFY_WS_URL", "ws://localhost:3000")
PREFERRED_SCHEMA = 21   # bropat bumps this as features are added; we cap at the server's max

if not SERIAL:
    sys.exit("Set EUFY_SERIAL in .env  (e.g.  EUFY_SERIAL=T8423XXXXXXXXXXXX)")


async def send(ws, command, **kwargs):
    """Send a JSON command and log it. Returns the messageId we generated."""
    msg_id = f"{command}-{uuid.uuid4().hex[:8]}"
    payload = {"messageId": msg_id, "command": command, **kwargs}
    await ws.send(json.dumps(payload))
    print(f"  ->  {command}  {kwargs}")
    return msg_id


async def keepalive():
    while True:
        try:
            print(f"\n[connect] {WS_URL}")
            async with websockets.connect(WS_URL, ping_interval=20, ping_timeout=20) as ws:
                # 1. First frame from server is the version/schema banner.
                hello = json.loads(await ws.recv())
                max_schema = hello.get("maxSchemaVersion", PREFERRED_SCHEMA)
                schema = min(PREFERRED_SCHEMA, max_schema)
                print(f"[server] driver={hello.get('driverVersion')}  "
                      f"server={hello.get('serverVersion')}  "
                      f"schemas=0..{max_schema}  (using {schema})")

                # 2. Pin schema, then start_listening to receive device events.
                await send(ws, "set_api_schema", schemaVersion=schema)
                await send(ws, "start_listening")

                # 3. Turn the rtspStream property ON. Without this, the next
                #    command "succeeds" but the camera doesn't actually serve.
                await send(ws, "device.set_property",
                           serialNumber=SERIAL, name="rtspStream", value=True)
                await asyncio.sleep(2)   # give the property a moment to propagate

                # 4. Kick the RTSP stream on.
                await send(ws, "device.start_rtsp_livestream", serialNumber=SERIAL)
                print(f"[ok] requested RTSP livestream for {SERIAL}\n")

                # 5. Listen forever. If the camera drops the stream, restart it.
                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type")

                    if mtype == "result" and not msg.get("success", True):
                        print(f"[err] {msg.get('messageId')} -> {msg.get('errorCode')}")

                    elif mtype == "event":
                        ev = msg.get("event", {})
                        if ev.get("serialNumber") != SERIAL:
                            continue
                        name = ev.get("event", "")
                        print(f"[event] {name}")
                        if "rtsp livestream stopped" in name:
                            print("        -> camera dropped stream; restarting in 3s")
                            await asyncio.sleep(3)
                            await send(ws, "device.start_rtsp_livestream",
                                       serialNumber=SERIAL)

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[disconnect] {e}\n[reconnect] in 5s...")
            await asyncio.sleep(5)


if __name__ == "__main__":
    try:
        asyncio.run(keepalive())
    except KeyboardInterrupt:
        print("\n[exit] bye")
