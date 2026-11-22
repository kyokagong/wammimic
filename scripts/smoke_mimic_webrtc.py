"""Verify video negotiation and changing simulation poses over native WebRTC."""

import argparse
import asyncio
from pathlib import Path

from aiohttp import ClientSession, ClientTimeout
from aiortc import RTCConfiguration, RTCPeerConnection, RTCSessionDescription


async def test(base, iteration, robot, mode, motion, capture, expect_five, upload_pico=None, release=None):
    release = release or f"joint-{mode}-{iteration}"
    pc = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    incoming = asyncio.get_running_loop().create_future()

    @pc.on("track")
    def receive(track):
        incoming.set_result(track)

    pc.addTransceiver("video", direction="recvonly")
    try:
        async with ClientSession(timeout=ClientTimeout(total=20), raise_for_status=True) as session:
            async def request(path, payload=None):
                async with session.request("POST" if payload is not None else "GET", base + path,
                                           json=payload) as response:
                    return await response.json()

            async def wait_ready():
                async with asyncio.timeout(240):
                    while True:
                        status = await request("/api/status")
                        if status["error"]:
                            raise RuntimeError(status["error"])
                        if (status["motion"] == motion and status["robot"] == robot
                                and status["release_id"] == release and status["frame_available"]
                                and not status["loading"]):
                            return status
                        await asyncio.sleep(1)

            async with session.get(base + "/") as response:
                page = await response.text()
                assert '<video id="frame"' in page
            async with session.get(base + "/app.js") as response:
                script = await response.text()
                assert "RTCPeerConnection" in script and "/api/offer" in script
                assert "/api/frame" not in script
            selection = dict(action="select", robot=robot, motion=motion, release=release)
            if upload_pico:
                async with session.post(base + "/api/recordings", data=upload_pico.read_bytes(),
                                        headers={"Content-Type": "application/x-ndjson",
                                                 "X-Filename": upload_pico.name}) as response:
                    selection["recording"] = (await response.json())["recording"]["id"]
            await request("/api/control", selection)
            ready = await wait_ready()
            if expect_five:
                assert ready["human_five_overlay"]
            elif mode == "motion":
                assert not ready.get("human_five_overlay", False)
            await pc.setLocalDescription(await pc.createOffer())
            local = pc.localDescription
            answer = await request("/api/offer", dict(sdp=local.sdp, type=local.type))
            await pc.setRemoteDescription(RTCSessionDescription(**answer))
            track = await asyncio.wait_for(incoming, 5)
            frame = await asyncio.wait_for(track.recv(), 20)
            assert frame.to_ndarray(format="rgb24").std() > 5
            await request("/api/control", {"action": "start"})
            try:
                for _ in range(18):
                    frame = await asyncio.wait_for(track.recv(), 8)
                status = await request("/api/status")
                encoders = status["webrtc"]["connections"]
                assert pc.connectionState == "connected" and status["steps"] >= 12, status
                assert any(peer["frames_sent"] >= 19 for peer in encoders), encoders
                if expect_five:
                    rgb = frame.to_ndarray(format="rgb24")
                    r, g, b = (rgb[..., i].astype(float) for i in range(3))
                    orange = (r > 150) & (g > .35 * r) & (g < .85 * r) & (b < .62 * g)
                    assert orange.sum() > 100, "Five-point tint is missing from the decoded frame"
                if capture:
                    frame.to_image().save(capture)
                print(dict(iteration=iteration, mode=mode, robot=robot, steps=status["steps"],
                           resolution=f"{frame.width}x{frame.height}", peers=encoders,
                           human_five_overlay=status.get("human_five_overlay", False)))
            finally:
                await request("/api/control", {"action": "pause"})
    finally:
        await pc.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--robot", choices=("q1", "bumi", "g1", "all"), default="q1")
    parser.add_argument("--mode", choices=("motion", "human"), default="motion")
    parser.add_argument("--motion", default="seed_flip_360_011__A416")
    parser.add_argument("--capture", type=Path)
    parser.add_argument("--expect-five", action="store_true")
    parser.add_argument("--upload-pico", type=Path)
    parser.add_argument("--release")
    args = parser.parse_args()
    asyncio.run(test(args.base, args.iteration, args.robot, args.mode,
                     args.motion, args.capture, args.expect_five, args.upload_pico, args.release))
