"""Stream per-viewer WAM evaluation replays via WebRTC."""

import argparse
import asyncio
from ipaddress import IPv4Address
import json
from pathlib import Path
from urllib.parse import urlsplit
from uuid import uuid4

from aiohttp import web
from aiortc import RTCConfiguration, RTCSessionDescription
from aiortc.contrib.media import MediaPlayer
from aiortc.sdp import SessionDescription
from preview.webrtc import PreviewPeerConnection


def mapped_sdp(sdp, media_host, media_ports):
    lines = []
    candidates = 0
    for line in sdp.splitlines():
        if line.startswith("a=candidate:"):
            fields = line.split()
            if fields[2].lower() != "udp" or int(fields[5]) not in media_ports:
                continue
            fields[4] = str(media_host)
            line = " ".join(fields)
            candidates += 1
        elif line.startswith("c=IN "):
            line = f"c=IN IP4 {media_host}"
        lines.append(line)
    if not candidates:
        raise ValueError("No ICE candidates use the mapped UDP ports")
    return "\r\n".join(lines) + "\r\n"


def create_app(directory, access_key, media_host, media_ports):
    manifest = json.loads((directory / "manifest.json").read_text())
    genie = manifest.get("kind") == "geniesim_q1"
    paths = []
    cases = []
    for case in manifest["cases"]:
        name = case["video"]
        path = directory / name
        if Path(name).name != name or path.suffix != ".mp4" or not path.is_file():
            raise ValueError(f"Invalid replay file: {name}")
        paths.append(path)
        if genie:
            keys = ("scenario", "seed", "mode", "success", "termination", "steps",
                    "min_wrist_distance_m", "hand_contact_steps", "button_max_travel_m")
            cases.append({key: case[key] for key in keys} | {"planning_calls": len(case["planning_calls"])})
        else:
            cases.append({key: case[key] for key in ("split", "selection", "ade_mm", "rotation_deg")} | {
                "scope": case["record"]["scope"], "episode": case["record"]["episode_index"],
                "frame": case["record"]["frame_index"],
            })
    catalog = {"step": manifest["step"], "cases": cases}
    if genie:
        catalog.update({key: manifest[key] for key in ("kind", "robot", "backend", "mimic_step")})
    else:
        catalog["metrics"] = manifest["metrics"]
    peers = {}

    async def close_peer(peer_id):
        entry = peers.pop(peer_id, None)
        if entry:
            pc, track, timeout = entry
            if timeout is not asyncio.current_task():
                timeout.cancel()
            track.stop()
            if pc.remoteDescription is not None:
                await pc.addIceCandidate(None)
            await pc.close()

    async def expire(peer_id):
        await asyncio.sleep(15)
        if peer_id in peers and peers[peer_id][0].connectionState != "connected":
            await close_peer(peer_id)

    async def shutdown(_):
        await asyncio.gather(*(close_peer(peer_id) for peer_id in tuple(peers)))

    @web.middleware
    async def headers(request, handler):
        response = await handler(request)
        response.headers.update({
            "Cache-Control": "no-store", "Referrer-Policy": "no-referrer",
            "X-Content-Type-Options": "nosniff", "X-Frame-Options": "DENY",
        })
        return response

    async def read_json(request):
        origin = request.headers.get("Origin")
        if origin and urlsplit(origin).netloc != request.host:
            raise web.HTTPForbidden()
        if request.content_type != "application/json":
            raise web.HTTPUnsupportedMediaType()
        try:
            data = await request.json()
        except (ValueError, UnicodeError) as error:
            raise web.HTTPBadRequest(reason="Invalid JSON") from error
        if not isinstance(data, dict):
            raise web.HTTPBadRequest()
        return data

    async def offer(request):
        data = await read_json(request)
        index = data.get("case")
        if type(index) is not int or not 0 <= index < len(paths) or not isinstance(data.get("sdp"), str):
            raise web.HTTPBadRequest(reason="Invalid case or SDP")
        sdp = data["sdp"]
        try:
            media = SessionDescription.parse(sdp).media
        except Exception as error:
            raise web.HTTPBadRequest(reason="Invalid SDP") from error
        if len(media) != 1 or media[0].kind != "video" or media[0].direction != "recvonly":
            raise web.HTTPBadRequest(reason="Offer must request one receive-only video stream")
        if len(peers) >= 4:
            raise web.HTTPServiceUnavailable(reason="There are four active viewers")
        pc = PreviewPeerConnection(RTCConfiguration(iceServers=[]))
        track = MediaPlayer(str(paths[index]), loop=True).video
        pc.addTrack(track)
        peer_id = uuid4().hex
        peers[peer_id] = pc, track, asyncio.create_task(expire(peer_id))

        @pc.on("connectionstatechange")
        @pc.on("iceconnectionstatechange")
        async def state_changed():
            if pc.connectionState == "connected":
                await pc.addIceCandidate(None)
            if pc.connectionState in ("failed", "closed") or pc.iceConnectionState == "failed":
                await close_peer(peer_id)

        # ICE discovers the authenticated return route through the mapped UDP sockets.
        sdp = "\r\n".join(line for line in sdp.splitlines()
                          if not line.startswith(("a=candidate:", "a=end-of-candidates"))) + "\r\n"
        try:
            async with asyncio.timeout(12):
                await pc.setRemoteDescription(RTCSessionDescription(sdp=sdp, type="offer"))
                await pc.setLocalDescription(await pc.createAnswer())
            return web.json_response({
                "type": "answer", "sdp": mapped_sdp(pc.localDescription.sdp, media_host, media_ports), "peer": peer_id,
            })
        except asyncio.CancelledError:
            await close_peer(peer_id)
            raise
        except (ValueError, TimeoutError) as error:
            await close_peer(peer_id)
            raise web.HTTPBadRequest(reason="WebRTC negotiation failed") from error
        except Exception:
            await close_peer(peer_id)
            raise

    async def disconnect(request):
        data = await read_json(request)
        if isinstance(data.get("peer"), str):
            await close_peer(data["peer"])
        return web.Response(status=204)

    async def metadata(_):
        return web.json_response(catalog)

    async def page(_):
        return web.FileResponse(Path(__file__).with_name("wam_preview.html"))

    app = web.Application(client_max_size=128 * 1024, middlewares=[headers])
    root = "/" + access_key + "/"
    app.router.add_get(root, page)
    app.router.add_get(root + "api/cases", metadata)
    app.router.add_post(root + "api/offer", offer)
    app.router.add_post(root + "api/close", disconnect)
    app.on_shutdown.append(shutdown)
    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--access-key-file", type=Path, required=True)
    parser.add_argument("--media-host", type=IPv4Address, required=True)
    parser.add_argument("--media-ports", type=int, nargs=2, default=(45970, 45989))
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    app = create_app(args.directory, args.access_key_file.read_text().strip(), args.media_host,
                     range(args.media_ports[0], args.media_ports[1] + 1))
    web.run_app(app, host="0.0.0.0", port=args.port, access_log=None)
