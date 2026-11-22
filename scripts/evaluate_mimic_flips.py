"""Re-evaluate the two held-out flips and measure actual feet clearance in simulation."""

import argparse
from collections import defaultdict
import csv
import json
from pathlib import Path
import re
from tempfile import TemporaryDirectory

import mujoco
import numpy as np
import torch
import warp as wp

from mjlab.utils.torch import configure_torch_backends
from morphmimic.env.env_cfg import COMMAND_NAME
from morphmimic.eval.benchmark import make_eval_env, run_benchmark
from morphmimic.eval.policy_loader import load_policy
from morphmimic.scripts.eval import checkpoint_identity, verify_checkpoint
from morphmimic.train.recovery import file_sha

FLIPS = {"seed_flip_180_R_006__A304", "seed_flip_360_011__A416"}
ROBOTS = ("q1", "bumi", "g1")
CLEARANCE_M = .002


class FlightTrace:
    def __init__(self, env):
        self.env = env
        self.command = env.command_manager.get_term(COMMAND_NAME)
        self.library = self.command.library
        self.rows = defaultdict(list)
        self.finished = set()
        self.step = env.step
        env.step = self.observe
        model = env.sim.mj_model
        self.geoms = []
        for g in range(model.ngeom):
            name = (model.geom(g).name or "").rsplit("/", 1)[-1]
            if not re.fullmatch(self.command.spec.foot_geom_regex, name):
                continue
            kind = mujoco.mjtGeom(model.geom_type[g])
            size = model.geom_size[g].copy()
            vertices = None
            if kind == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = model.geom_dataid[g]
                start = model.mesh_vertadr[mesh]
                vertices = model.mesh_vert[start:start + model.mesh_vertnum[mesh]]
            self.geoms.append((g, kind, size, vertices))
        if not self.geoms:
            raise ValueError("Missing native foot collision geometry")
        self.floor = torch.as_tensor(model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE, device=env.device)
        self.geom_ids = torch.tensor([geom[0] for geom in self.geoms], device=env.device)
        self.origins_z = env.scene.env_origins[:, 2].cpu().numpy()

    def sole_clearance(self, data):
        xyz = wp.to_torch(data.geom_xpos).index_select(1, self.geom_ids).cpu().numpy()
        rotation = wp.to_torch(data.geom_xmat).index_select(1, self.geom_ids).cpu().numpy()
        lowest = []
        for k, (_, kind, size, vertices) in enumerate(self.geoms):
            z = rotation[:, k, 2]
            if kind == mujoco.mjtGeom.mjGEOM_BOX:
                support = np.abs(z) @ size
            elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                support = size[0]
            elif kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                support = size[0] + np.abs(z[:, 2]) * size[1]
            elif kind == mujoco.mjtGeom.mjGEOM_CYLINDER:
                support = size[0] * np.linalg.norm(z[:, :2], axis=-1) + np.abs(z[:, 2]) * size[1]
            elif kind == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
                support = np.linalg.norm(z * size, axis=-1)
            elif kind == mujoco.mjtGeom.mjGEOM_MESH:
                support = -(z @ vertices.T).min(axis=-1)
            else:
                raise ValueError(f"Unsupported foot geometry {kind}")
            lowest.append(xyz[:, k, 2] - support - self.origins_z)
        return np.min(lowest, axis=0)

    def observe(self, action):
        frames = self.command.global_frame.clone()
        result = self.step(action)
        _, _, terminated, truncated, _ = result
        data = self.env.sim.wp_data
        count = int(wp.to_torch(data.nacon)[0])
        contact = data.contact
        ids = wp.to_torch(contact.geom)[:count].long()
        valid = (ids >= 0).all(-1) & (wp.to_torch(contact.dist)[:count] <= CLEARANCE_M)
        has_floor = self.floor[ids.clamp_min(0)].any(-1)
        worlds = wp.to_torch(contact.worldid)[:count][valid & has_floor].long()
        contacts = torch.bincount(worlds, minlength=self.env.num_envs).tolist()
        clearance = self.sole_clearance(data)
        heights = (self.command.robot_body_pos_w[:, 0, 2] - self.env.scene.env_origins[:, 2]).tolist()
        clip_ids = self.library.frame_clip[frames].tolist()
        term = terminated.tolist()
        trunc = truncated.tolist()
        manager = self.env.termination_manager
        causes = {name: manager.get_term(name).tolist() for name in manager.active_terms}
        for i, clip_idx in enumerate(clip_ids):
            name = self.library.clip_names[clip_idx]
            if name in self.finished:
                continue
            self.rows[name].append(dict(
                frame=int(frames[i] - self.library.clip_start[clip_idx]),
                sole_clearance_m=float(clearance[i]), root_height_m=heights[i], floor_contacts=contacts[i],
                termination_terms=[key for key, flags in causes.items() if term[i] and flags[i]],
            ))
            if term[i] or trunc[i]:
                self.finished.add(name)
        return result

    def metrics(self, name, failed):
        rows = self.rows[name]
        assert rows and name in self.finished
        flight = longest = 0
        clearance = 0.
        for row in rows:
            flying = row["floor_contacts"] == 0 and row["sole_clearance_m"] > CLEARANCE_M
            flight = flight + 1 if flying else 0
            longest = max(longest, flight)
            if flying:
                clearance = max(clearance, row["sole_clearance_m"])
        terms = rows[-1]["termination_terms"]
        if failed and not terms:
            raise ValueError(f"Missing native termination reason for {name}")
        first = self.library.clip_start[self.library.clip_names.index(name)]
        standing_z = float(self.library.body_pos_w[first, 0, 2])
        return dict(max_airborne_clearance_m=clearance,
                    max_root_rise_m=max(row["root_height_m"] for row in rows) - standing_z,
                    longest_flight_s=round(longest * self.env.step_dt, 4),
                    termination_terms=terms)


def evaluate(run, robot):
    config = json.loads((run / "config.json").read_text())
    path = Path(config["prepared_manifest"])
    manifest_sha = file_sha(path)
    manifest = json.loads(path.read_text())
    selected = [r for r in manifest["clips"] if r["robot"] == robot and r["split"] == "val" and r["name"] in FLIPS]
    if len(selected) != 2 or {r["name"] for r in selected} != FLIPS:
        raise ValueError("Expected the same two held-out flips")
    for row in selected:
        if (file_sha(row["output"]) != row["output_sha256"] or not row["human_available"]
                or file_sha(row["human_sidecar"]) != row["human_sha256"]):
            raise ValueError(f"Held-out reference identity differs for {row['name']}")
    checkpoint = run / robot / f"ckpt_{config['iterations']}.pt"
    identity = checkpoint_identity(checkpoint)
    result, traces = [], {}
    for interface in ("motion", "human"):
        policy, meta = load_policy(checkpoint, "cuda:0", interface=interface, human_points="five",
                                   kp_horizon=1, goal_anchor=config["goal_anchor"])
        with TemporaryDirectory(prefix="mimic-heldout-flips-") as folder:
            subset = Path(folder) / "subset.json"
            subset.write_text(json.dumps(dict(root="/", clips=[row["output"] for row in selected])))
            env = make_eval_env(str(subset), 2, "cuda:0", robot=robot, library_dtype=config["library_dtype"],
                                human_root=config["human_root"], goal_anchor=config["goal_anchor"],
                                ref_feedback=config["ref_feedback"])
            try:
                trace = FlightTrace(env)
                scored = run_benchmark(env, policy, meta.hist_len)
                if set(scored["name"]) != FLIPS or env.physics_diagnostics.report()["overflow_bits"]:
                    raise ValueError("Flip evaluation inventory or native physics differs")
                for row in scored.to_dict("records"):
                    name = row["name"]
                    result.append(dict(iteration=meta.iteration, robot=robot, interface=interface,
                                       clip=name, steps=row["steps"], fail_frame=row["fail_frame"],
                                       terminated=row["failed"], global_success=bool(row["global_success"]),
                                       **trace.metrics(name, row["failed"])))
                traces[interface] = dict(trace.rows)
            finally:
                env.close()
    verify_checkpoint(identity)
    return dict(checkpoint=identity, training_manifest_sha256=manifest_sha,
                contact_epsilon_m=CLEARANCE_M, simulation_step_s=.02, results=result, traces=traces)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", choices=ROBOTS)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    configure_torch_backends(allow_tf32=False)
    with torch.no_grad():
        for robot in (args.robot,) if args.robot else ROBOTS:
            report = evaluate(args.run.resolve(strict=True), robot)
            name = f"{robot}-{report['results'][0]['iteration']}"
            path = args.output / f"{name}.json"
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(report, ensure_ascii=False, allow_nan=False))
            with (args.output / f"{name}.csv").open("w", newline="") as stream:
                rows = report["results"]
                writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
                writer.writeheader()
                writer.writerows(rows)
            temp.replace(path)
            print(json.dumps(report["results"]), flush=True)


if __name__ == "__main__":
    main()
