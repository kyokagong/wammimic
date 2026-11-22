"""Serve fixed Mimic releases through BlinkMorphMimic's native WebRTC Preview."""

import argparse
import asyncio
import copy
import json
from pathlib import Path
from threading import Thread

from aiohttp import web

from preview import server
from preview.comparison import ComparisonSession
from preview.webrtc import create_app


def display_paths(value):
    if isinstance(value, dict):
        return {key: display_paths(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [display_paths(item) for item in value]
    if isinstance(value, str) and value.startswith("/"):
        return Path(value).name
    return value


class PublicPreview(server.Preview):
    def snapshot(self):
        return display_paths(super().snapshot())


def prepare_release(run):
    config = json.loads((run / "config.json").read_text())
    receipt = json.loads((run / "freeze.json").read_text())
    iteration = config["iterations"]
    interval = config["ckpt_every"]
    keep = config["keep_checkpoints"]
    if any(type(value) is not int or value <= 0 for value in (iteration, interval, keep)):
        raise ValueError("Expected positive fixed training and checkpoint settings")
    steps = sorted(set(range(interval, iteration + 1, interval)) | {iteration})[-keep:]
    if (iteration, interval, keep) == (12_000, 1_000, 4):
        steps = [11_000, 12_000]
    required = {f"ckpt_{step}.pt" for step in steps}
    manifest = Path(config["prepared_manifest"])
    checks = {}
    for robot in server.ROBOTS:
        folder = run / robot
        retained = {path.name for path in folder.glob("ckpt_*.pt")}
        allowed = required | ({"ckpt_10000.pt"} if iteration == 12_000 and interval == 1_000 else set())
        if not required <= retained <= allowed:
            raise ValueError(f"{robot} does not have the expected checkpoint files")
        checkpoint = server.FixedCheckpoint.pin(folder / f"ckpt_{iteration}.pt", robot)
        checks[robot] = {"path": str(checkpoint.path), "sha256": checkpoint.sha256}
    manifest_ref = {"path": str(manifest), "sha256": receipt["data_manifest_sha256"]}
    releases = [dict(
        id=f"joint-{interface}-{iteration}", label=f"{iteration:,} · Global + Ref. · {label}", interface=interface,
        data_root=str(manifest.parent), checkpoints=checks, parent_iteration=iteration, human_updates=iteration,
        training_manifest=manifest_ref, reference_manifest=manifest_ref,
        recovery_windows={"path": config["recovery_windows"], "sha256": config["recovery_windows_sha256"]},
    ) for interface, label in (("motion", "Motion"), ("human", "Human 五点 + 机型参考"))]
    if json.loads(manifest.read_text()).get("five_point_root"):
        human = releases[1]
        human["five_point_source"] = "original_human"
        human["label"] = f"{iteration:,} · 原始人体五点 · H{config['kp_fixed_horizon']}"
        for source, label in (("robot_reference", "机器人参考五点"), ("mixed", "混合来源五点")):
            entry = copy.deepcopy(human)
            entry.update(id=f"joint-{source.replace('_', '-')}-{iteration}", five_point_source=source,
                         label=f"{iteration:,} · {label} · H{config['kp_fixed_horizon']}")
            releases.append(entry)
    return releases


def release_path(run):
    path = run / "preview-releases.json"
    if not path.is_file():
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(prepare_release(run), indent=2, ensure_ascii=False))
        temporary.replace(path)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run", type=Path)
    source.add_argument("--releases", type=Path)
    parser.add_argument("--compare-run", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=31777)
    parser.add_argument("--public-host")
    parser.add_argument("--nvenc", action="store_true")
    parser.add_argument("--pico-data-root", type=Path)
    args = parser.parse_args()

    releases = server.load_release_catalogs(release_path(args.run) if args.run else args.releases)
    if args.compare_run:
        for name, entry in server.load_release_catalogs(release_path(args.compare_run)).items():
            if name in releases:
                raise ValueError("Duplicate comparison release")
            releases[name] = entry
    catalog = next(iter(releases.values()))["catalog"]
    preview = PublicPreview(catalog, args.device, releases=releases, comparison_factory=ComparisonSession)
    if args.pico_data_root:
        from preview.pico import PicoRecordings
        human = next(entry["catalog"] for entry in releases.values()
                     if entry["catalog"].interface == "human")
        templates = {}
        for robot in server.ROBOTS:
            motion = human.select(robot, "dance1_subject1")
            templates[robot] = dict(motion=motion, human=Path(human.human_root) / robot / motion.name)
        preview.recordings = PicoRecordings(args.pico_data_root, templates)
    worker = Thread(target=preview.run, daemon=True)
    app = create_app(preview, public_host=args.public_host, nvenc=args.nvenc)

    async def start(_):
        worker.start()

    async def stop(_):
        preview.stop.set()
        await asyncio.to_thread(worker.join, 10)

    app.on_startup.append(start)
    app.on_cleanup.append(stop)
    web.run_app(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
