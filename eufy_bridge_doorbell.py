"""
eufy_bridge_doorbell.py
-----------------------
Twin of eufy_bridge.py but for the T8200 doorbell. Reads its own env
variables so the camera bridge and the doorbell bridge can run side by
side without colliding on UDP port or device serial.

Setup additions to .env:
    EUFY_DOORBELL_SERIAL=T8200N0020281CF1
    DOORBELL_URL=udp://127.0.0.1:5001?fifo_size=5000000&overrun_nonfatal=1

BATTERY WARNING:
    The T8200 is a battery-powered doorbell. Continuous P2P livestreaming
    will drain it significantly faster than its normal duty cycle. Plan
    for short experimental sessions and charge between tests.

Run order (after eufy-security-server is already up on :3000):
    Window 4:  py -3.12 eufy_bridge_doorbell.py
    Window 5:  py -3.12 doorbell_node.py
"""

import asyncio
import base64
import json
import os
import subprocess
import sys
import uuid

from dotenv import load_dotenv
import websockets

load_dotenv()

SERIAL = os.getenv("EUFY_DOORBELL_SERIAL")
WS_URL = os.getenv("EUFY_WS_URL", "ws://localhost:3000")
UDP_OUT = os.getenv("EUFY_DOORBELL_UDP_OUT", "udp://127.0.0.1:5001?pkt_size=1316")
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
PREFERRED_SCHEMA = 21

if not SERIAL:
    sys.exit("Set EUFY_DOORBELL_SERIAL in .env (e.g. T8200N0020281CF1)")


def start_ffmpeg(codec):
    args = [
        FFMPEG,
        "-hide_banner", "-loglevel", "warning",
        "-fflags", "+genpts+discardcorrupt+nobuffer",
        "-flags", "low_delay",
        "-probesize", "32",
        "-analyzeduration", "0",
        "-f", codec,
        "-i", "pipe:0",
        "-c", "copy",
        "-bsf:v", "dump_extra=freq=keyframe",
        "-mpegts_flags", "+resend_headers+pat_pmt_at_frames",
        "-f", "mpegts",
        UDP_OUT,
    ]
    print(f"[ffmpeg] launching -> {UDP_OUT}")
    return subprocess.Popen(
        args, stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, bufsize=0,
    )


def extract_bytes(buf):
    if isinstance(buf, str):
        try:
            return base64.b64decode(buf)
        except Exception:
            return None
    if isinstance(buf, dict):
        data = buf.get("data")
        if isinstance(data, list):
            return bytes(data)
        if isinstance(data, str):
            try:
                return base64.b64decode(data)
            except Exception:
                return None
    return None


async def send(ws, command, **kwargs):
    msg_id = f"{command}-{uuid.uuid4().hex[:8]}"
    payload = {"messageId": msg_id, "command": command, **kwargs}
    await ws.send(json.dumps(payload))
    print(f"  ->  {command}  {kwargs}")
    return msg_id


async def bridge():
    while True:
        ffmpeg_proc = None
        chunks_received = 0
        bytes_received = 0
        try:
            print(f"\n[doorbell-connect] {WS_URL}")
            async with websockets.connect(
                WS_URL, ping_interval=20, ping_timeout=20, max_size=None,
            ) as ws:
                hello = json.loads(await ws.recv())
                schema = min(PREFERRED_SCHEMA, hello.get("maxSchemaVersion", PREFERRED_SCHEMA))
                print(f"[doorbell-server] using schema {schema}")

                await send(ws, "set_api_schema", schemaVersion=schema)
                await send(ws, "start_listening")
                await asyncio.sleep(1)

                await send(ws, "device.start_livestream", serialNumber=SERIAL)

                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type")

                    if mtype == "result" and not msg.get("success", True):
                        print(f"[doorbell-err] {msg.get('messageId')} -> {msg.get('errorCode')}")
                        continue
                    if mtype != "event":
                        continue

                    ev = msg.get("event", {})
                    if ev.get("serialNumber") != SERIAL:
                        continue
                    name = ev.get("event", "")

                    if name == "livestream started":
                        print("[doorbell-event] livestream started")

                    elif name == "livestream stopped":
                        print("[doorbell-event] livestream stopped — restarting in 3s")
                        if ffmpeg_proc:
                            try:
                                ffmpeg_proc.stdin.close()
                                ffmpeg_proc.terminate()
                            except Exception:
                                pass
                            ffmpeg_proc = None
                        await asyncio.sleep(3)
                        await send(ws, "device.start_livestream", serialNumber=SERIAL)

                    elif name == "livestream video data":
                        if ffmpeg_proc is None:
                            md = ev.get("metadata") or {}
                            codec_raw = (md.get("videoCodec") or "h264").lower()
                            codec = {"h264": "h264", "h265": "hevc",
                                     "hevc": "hevc"}.get(codec_raw, "h264")
                            print(f"[doorbell-video] codec={codec_raw!r} -> ffmpeg -f {codec}")
                            ffmpeg_proc = start_ffmpeg(codec)

                        chunk = extract_bytes(ev.get("buffer"))
                        if not chunk:
                            continue
                        try:
                            ffmpeg_proc.stdin.write(chunk)
                            chunks_received += 1
                            bytes_received += len(chunk)
                            if chunks_received % 200 == 0:
                                print(f"[doorbell-stream] {chunks_received:>5} chunks, "
                                      f"{bytes_received/1024:.0f} KB")
                        except (BrokenPipeError, ValueError, OSError) as e:
                            print(f"[doorbell-ffmpeg] pipe broke ({e}); respawning")
                            try:
                                ffmpeg_proc.terminate()
                            except Exception:
                                pass
                            ffmpeg_proc = None

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[doorbell-disconnect] {e}; reconnect in 5s")
            await asyncio.sleep(5)
        finally:
            if ffmpeg_proc:
                try:
                    ffmpeg_proc.stdin.close()
                    ffmpeg_proc.terminate()
                except Exception:
                    pass


if __name__ == "__main__":
    try:
        asyncio.run(bridge())
    except KeyboardInterrupt:
        print("\n[doorbell-exit]")
