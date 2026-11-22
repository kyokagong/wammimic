"""Check the final checkpoint on the same physical frame in eval and Preview."""

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import torch
from mjlab.utils.lab_api.math import quat_apply_inverse

from morphmimic.constants import ROOT_IDX
from morphmimic.env.env_cfg import COMMAND_NAME
from morphmimic.env.observations import wm_state
from morphmimic.eval.benchmark import make_eval_env
from morphmimic.eval.policy_loader import load_policy
from morphmimic.model.ref_feedback import REF_BODIES, REF_FRAMES, feedback_frame, global_ref_feedback
from morphmimic.model.state import join_state, split_state
from morphmimic.train.collector import goal_groups
from morphmimic.train.obs_history import ObsHistory
from morphmimic.train.run_identity import ManifestIdentity
from preview.server import FixedCheckpoint, PolicySession
from preview.shared_policy import SharedPolicy


@torch.no_grad()
def verify(run, device, iteration=None):
    config = json.loads((run / "config.json").read_text())
    iteration = iteration or config["iterations"]
    identity = ManifestIdentity(config["prepared_manifest"])
    manifest = json.loads(identity.path.read_text())
    results = {}
    checkpoints = {}
    interface_cases = {"motion": {}, "human": {}}
    for robot in config["robots"]:
        result = {}
        row = next(row for row in manifest["clips"]
                   if row["robot"] == robot and row["split"] == "val" and row["human_available"])
        path = Path(row["output"])
        checkpoint = FixedCheckpoint.pin(run / robot / f"ckpt_{iteration}.pt", robot)
        checkpoints[robot] = checkpoint
        for interface in ("motion", "human"):
            with TemporaryDirectory(prefix="mimic-feedback-check-") as temporary:
                subset = Path(temporary) / "subset.yaml"
                subset.write_text(json.dumps({"root": str(path.parent), "clips": [path.name]}))
                policy, meta = load_policy(checkpoint.path, device, interface=interface,
                                           human_points="five", kp_horizon=1, goal_anchor=config["goal_anchor"])
                if not meta.train_cfg["ref_feedback"]:
                    raise ValueError("Mimic checkpoint does not enable Global + Ref.")
                session = PolicySession(robot, path, checkpoint, device, render=False, interface=interface,
                                        human_root=config["human_root"], paired_manifest_sha256=identity.sha256,
                                        training_manifest_sha256=identity.sha256)
                try:
                    env = make_eval_env(str(subset), 1, device, robot=robot, ref_feedback=True,
                                        library_dtype=config["library_dtype"], human_root=config["human_root"],
                                        goal_anchor=config["goal_anchor"])
                    try:
                        command = env.command_manager.get_term(COMMAND_NAME)
                        ids = torch.zeros(1, dtype=torch.long, device=device)
                        command.assign_clips(ids, ids)
                        obs, _ = env.reset(env_ids=ids)
                        # Verify residuals and actions on several consecutive physical frames.
                        delta = None
                        world_feedback_diff = same_input_diff = 0.
                        for step in range(6):
                            frame = feedback_frame(command.library, command.global_frame)
                            if int(frame.item()) != int(command.library.clip_start[0]) + step:
                                raise ValueError("Reference clock differs from the current physics tick")
                            reference = command.library.future(frame, REF_FRAMES)
                            remaining = command.library.remaining(frame)
                            goals = {name: obs[name] for name in goal_groups(obs)}
                            goals["human_joint_valid"] = command.library.human_valid_at(frame)
                            history = session.history.flat()
                            preview_action = session.policy(history, goals)
                            action = policy(history, goals)
                            torch.testing.assert_close(goals["ref_feedback"],
                                                       global_ref_feedback(obs["state"], reference, remaining))
                            preview_command = session.command
                            preview_frame = feedback_frame(preview_command.library, preview_command.global_frame)
                            preview_ref = preview_command.library.future(preview_frame, REF_FRAMES)
                            preview_remaining = preview_command.library.remaining(preview_frame)
                            torch.testing.assert_close(session.goals["ref_feedback"], global_ref_feedback(
                                wm_state(session.env, COMMAND_NAME), preview_ref, preview_remaining))
                            # Independently stepped Warp worlds may have floating-point drift.
                            torch.testing.assert_close(goals["ref_feedback"], session.goals["ref_feedback"],
                                                       atol=2e-4, rtol=1e-5)
                            world_feedback_diff = max(world_feedback_diff, float(
                                (goals["ref_feedback"] - session.goals["ref_feedback"]).abs().max()))
                            torch.testing.assert_close(action, preview_action, atol=2e-5, rtol=1e-5)
                            same_input_diff = max(same_input_diff, float((action - preview_action).abs().max()))
                            if step == 1:
                                state = split_state(obs["state"])
                                shift = obs["state"].new_tensor([[.7, 0., 0.]])
                                moved = join_state(state._replace(body_pos=state.body_pos + shift[:, None]))
                                feedback = global_ref_feedback(moved, reference, remaining)
                                n = REF_FRAMES * len(REF_BODIES)
                                before = goals["ref_feedback"][:, :9*n].reshape(1, REF_FRAMES, len(REF_BODIES), 9)
                                after = feedback[:, :9*n].reshape_as(before)
                                expected = -quat_apply_inverse(state.body_quat[:, ROOT_IDX], shift)
                                torch.testing.assert_close(after[..., :3] - before[..., :3],
                                                           expected[:, None, None].expand_as(after[..., :3]),
                                                           atol=1e-5, rtol=1e-5)
                                changed_action = policy(history, dict(goals, ref_feedback=feedback))
                                delta = float((changed_action - action).abs().max())
                                if not delta > 1e-5:
                                    raise ValueError(f"{robot}/{interface}: trained Actor ignores feedback")
                                case = SimpleNamespace(history=ObsHistory.from_batch(session.history.buf.clone()),
                                                       goals={key: value.clone() for key, value in goals.items()},
                                                       env=SimpleNamespace(device=device))
                                interface_cases[interface][robot] = case, action.clone()
                            session.step(action)
                            obs, _, terminated, truncated, _ = env.step(action)
                            if bool((terminated | truncated).any()):
                                raise ValueError(f"{robot}/{interface}: premature termination in parity check")
                        result[interface] = {"clip": row["name"], "matched_frames": 6,
                                             "root_shift_m": .7, "max_action_change": delta,
                                             "max_world_feedback_difference": world_feedback_diff,
                                             "max_same_input_action_difference": same_input_diff}
                    finally:
                        env.close()
                finally:
                    session.close()
        results[robot] = result
        print(f"{robot}: trained policy feedback and Preview parity passed", flush=True)
    for interface, cases in interface_cases.items():
        shared = SharedPolicy(checkpoints, device, interface=interface,
                              paired_manifest_sha256=identity.sha256, training_manifest_sha256=identity.sha256)
        batched = shared({robot: case for robot, (case, _) in cases.items()})
        for robot, (_, action) in cases.items():
            torch.testing.assert_close(batched[robot], action, atol=2e-5, rtol=1e-5)
            results[robot][interface]["max_batched_action_difference"] = float(
                (batched[robot] - action).abs().max())
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iteration", type=int)
    args = parser.parse_args()
    results = verify(args.run, args.device, args.iteration)
    filename = f"feedback-verification-{args.iteration}.json" if args.iteration else "feedback-verification.json"
    output = args.run / filename
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2))
    temporary.replace(output)


if __name__ == "__main__":
    main()
