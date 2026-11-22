"""Exercise the Preview API and renderer with both final-checkpoint interfaces."""

import argparse
import json
import time
from urllib.error import URLError
from urllib.request import Request, urlopen


def request(base, path, payload=None):
    body = json.dumps(payload).encode() if payload is not None else None
    headers = {"Content-Type": "application/json"} if body is not None else {}
    with urlopen(Request(base + path, data=body, headers=headers), timeout=10) as response:
        return response.read()


def status_until(base, matches, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            status = json.loads(request(base, "/api/status"))
        except URLError:
            time.sleep(1)
            continue
        if status["error"]:
            raise RuntimeError(status["error"])
        if matches(status):
            return status
        time.sleep(1)
    raise TimeoutError("Preview did not reach the expected state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("base")
    parser.add_argument("--iteration", type=int, required=True)
    parser.add_argument("--ready-only", action="store_true")
    args = parser.parse_args()

    release_ids = {f"joint-{interface}-{args.iteration}" for interface in ("motion", "human")}
    status_until(args.base, lambda status: release_ids <= {
        release["id"] for release in status["releases"] if "Global + Ref." in release["label"]
    }, 120)
    if args.ready_only:
        return
    for interface, robot in (("motion", "q1"), ("human", "all")):
        release = f"joint-{interface}-{args.iteration}"
        query = f"/api/motions?robot={robot}&release={release}&q=dance2_subject3"
        motion, = json.loads(request(args.base, query))["motions"]
        request(args.base, "/api/control", dict(action="select", robot=robot, motion=motion, release=release))
        status_until(args.base, lambda status: status["robot"] == robot and status["interface"] == interface
                     and status["frame_available"] and not status["loading"], 300)
        request(args.base, "/api/control", {"action": "start"})
        status = status_until(args.base, lambda status: status["steps"] >= 12 and status["frame_available"], 90)
        request(args.base, "/api/control", {"action": "pause"})
        assert not status["failed"] and request(args.base, "/api/frame").startswith(b"\xff\xd8")
        print(f"{interface}/{robot}: policy physics and JPEG confirmed", flush=True)


if __name__ == "__main__":
    main()
