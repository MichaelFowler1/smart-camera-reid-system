"""
eufy_bridge.py
--------------
Replaces eufy_keepalive.py. Connects to the eufy-security-server you have
running on ws://localhost:3000, requests a continuous P2P livestream from
the S330, decodes the H.264 chunks coming back over the WebSocket, and
republishes them as a local UDP MPEG-TS stream. reid_node.py reads that
local UDP stream instead of the camera's flaky on-demand RTSP URL.

Pipeline:
    Eufy camera --P2P--> eufy-security-server --WS--> THIS SCRIPT
       --ffmpeg stdin--> ffmpeg --UDP/MPEG-TS--> reid_node.py

Why this exists: The S330's firmware doesn't expose direct RTSP-on-demand
via the P2P API. P2P livestream IS supported, though, so we use that path
and rebuild an OpenCV-friendly stream on the way out.

Setup (one-time):
  1. Have eufy-security-server running on ws://localhost:3000 (window 1).
  2. Install ffmpeg for Windows. Easiest: in PowerShell run:
         winget install Gyan.FFmpeg
     Then close and reopen PowerShell, and verify:
         ffmpeg -version
     If that prints version info, you're set.
  3. pip install websockets python-dotenv         (you already have these)
  4. .env must contain:
         EUFY_SERIAL=T8423N10221614D3
         CAMERA_URL=udp://127.0.0.1:5000?fifo_size=5000000&overrun_nonfatal=1
     (was rtsp://...; the udp:// URL is what reid_node.py reads now)

Run order (three PowerShell windows):
    Window 1:  eufy-security-server --config config.json
    Window 2:  python eufy_bridge.py
    Window 3:  python reid_node.py
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

SERIAL = os.getenv("EUFY_SERIAL")
WS_URL = os.getenv("EUFY_WS_URL", "ws://localhost:3000")
UDP_OUT = os.getenv("EUFY_UDP_OUT", "udp://127.0.0.1:5000?pkt_size=1316")
FFMPEG = os.getenv("FFMPEG_BIN", "ffmpeg")
PREFERRED_SCHEMA = 21

if not SERIAL:
    sys.exit("Set EUFY_SERIAL in .env (e.g. EUFY_SERIAL=T8423N10221614D3)")


def start_ffmpeg(codec):
    """
    Spawn ffmpeg as a child process. It reads raw H.264/H.265 NAL units
    from stdin, wraps them in MPEG-TS, and pushes to a local UDP port.
    `-c copy` means no transcoding — fast, low CPU, no quality loss.
    """
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
        # Re-inject SPS/PPS in front of every keyframe so a decoder that
        # joins the UDP stream mid-flight (like reid_node.py's OpenCV
        # starting up) can sync within one GOP instead of failing forever.
        "-bsf:v", "dump_extra=freq=keyframe",
        # Same idea at the MPEG-TS layer: re-emit PAT/PMT regularly.
        "-mpegts_flags", "+resend_headers+pat_pmt_at_frames",
        "-f", "mpegts",
        UDP_OUT,
    ]
    print(f"[ffmpeg] launching -> {UDP_OUT}")
    return subprocess.Popen(
        args,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        bufsize=0,
    )


def extract_bytes(buf):
    """
    eufy-security-ws sends each chunk in one of two shapes depending on
    schema version. Handle both:
        1. base64 string
        2. Node-style Buffer dict: {"type": "Buffer", "data": [0, 1, 2, ...]}
    """
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
            print(f"\n[connect] {WS_URL}")
            async with websockets.connect(
                WS_URL,
                ping_interval=20,
                ping_timeout=20,
                max_size=None,  # don't cap message size; video chunks can be big
            ) as ws:
                hello = json.loads(await ws.recv())
                schema = min(PREFERRED_SCHEMA, hello.get("maxSchemaVersion", PREFERRED_SCHEMA))
                print(f"[server] driver={hello.get('driverVersion')}  "
                      f"server={hello.get('serverVersion')}  using schema {schema}")

                await send(ws, "set_api_schema", schemaVersion=schema)
                await send(ws, "start_listening")
                await asyncio.sleep(1)   # let server enumerate devices

                await send(ws, "device.start_livestream", serialNumber=SERIAL)

                async for raw in ws:
                    msg = json.loads(raw)
                    mtype = msg.get("type")

                    if mtype == "result" and not msg.get("success", True):
                        print(f"[err] {msg.get('messageId')} -> {msg.get('errorCode')}")
                        continue

                    if mtype != "event":
                        continue

                    ev = msg.get("event", {})
                    if ev.get("serialNumber") != SERIAL:
                        continue
                    name = ev.get("event", "")

                    if name == "livestream started":
                        print("[event] livestream started")

                    elif name == "livestream stopped":
                        print("[event] livestream stopped — restarting in 3s")
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
                        # Lazily spawn ffmpeg on the first video chunk so we
                        # know the codec the camera is actually sending.
                        if ffmpeg_proc is None:
                            md = ev.get("metadata") or {}
                            codec_raw = (md.get("videoCodec") or "h264").lower()
                            codec = {"h264": "h264", "h265": "hevc",
                                     "hevc": "hevc"}.get(codec_raw, "h264")
                            print(f"[video] codec reported as {codec_raw!r} "
                                  f"-> ffmpeg -f {codec}")
                            ffmpeg_proc = start_ffmpeg(codec)

                        chunk = extract_bytes(ev.get("buffer"))
                        if not chunk:
                            continue

                        try:
                            ffmpeg_proc.stdin.write(chunk)
                            chunks_received += 1
                            bytes_received += len(chunk)
                            if chunks_received % 200 == 0:
                                print(f"[stream] {chunks_received:>5} chunks, "
                                      f"{bytes_received/1024:.0f} KB sent to ffmpeg")
                        except (BrokenPipeError, ValueError, OSError) as e:
                            print(f"[ffmpeg] pipe broke ({e}); will respawn")
                            try:
                                ffmpeg_proc.terminate()
                            except Exception:
                                pass
                            ffmpeg_proc = None

                    # "livestream audio data" is ignored — RT-DETR only needs video.

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"[disconnect] {e}\n[reconnect] in 5s...")
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
        print("\n[exit] bye")