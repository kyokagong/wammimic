"""Export physically successful Q1 button demonstrations without invented EE joints."""
import io
import json
import shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from PIL import Image

from .common import sha256, write_json
from .human_ego import human_features, human_history
from .openhlm import encode_video

DATASET = "wam_lerobot_v2/q1_stub"


def import_sim(raw_root, output):
    raw_root, output = Path(raw_root), Path(output)
    dataset = output / DATASET
    episodes, tasks, action_values, state_values = [], {}, [], []
    for episode in sorted((raw_root / "episodes").glob("*/episode.json")):
        row = json.loads(episode.read_text())
        if not row["accepted"] or not row["physical_button_success"]:raise ValueError("Unsuccessful simulation in accepted folder")
        is_avoid = row.get("task") == "avoid_obstacle_then_press_button"
        min_t, min_w = (0.60, 0.20) if is_avoid else (0.80, 0.40)
        if row.get('target_in_frustum_fraction',0)<min_t or row.get('target_and_wrist_in_frustum_fraction',0)<min_w:
            raise ValueError('Simulation failed target/working-wrist framing gate')
        if is_avoid and row.get("obstacle_collision_steps", 1) != 0:
            raise ValueError('Obstacle-avoidance episode collided with floor obstacle')
        source = episode.parent / "trajectory.npz"
        if sha256(source) != row["trajectory_sha256"]:raise ValueError("Changed simulation trajectory")
        index = len(episodes)
        receipt = dataset / f"receipts/episode_{index:06d}.json"
        task = tasks.setdefault(row["instruction"], len(tasks))
        if receipt.exists():
            saved = json.loads(receipt.read_text())
            if saved["source_sha256"] != sha256(episode):raise ValueError("Changed simulation source on resume")
            if any(sha256(output/p) != digest for p,digest in saved["outputs"].items()):raise ValueError("Changed simulation export")
            episodes.append(saved)
            continue
        with np.load(source, allow_pickle=False) as data:
            poses, stamps, frames = data["poses_world"], data["time_s"], data["rgb_frame_index"]
            if not np.allclose(np.diff(stamps), .02, atol=1e-8) or not np.array_equal(frames, np.arange(0,len(stamps),5)):
                raise ValueError("Simulation 50 Hz state / 10 Hz camera mismatch")
            if data["joint_pos"].shape != (len(stamps),len(row["joint_names_native"])):
                raise ValueError("Simulation native joint identities differ")
            count = len(frames)
            projected=data['target_wrist_uv_depth']
            if projected.shape!=(count,2,3):raise ValueError('Missing camera projection measurements')
            visible=((projected[...,:2]>=.03)&(projected[...,:2]<=.97)).all(-1)&(projected[...,2]>0)
            if not np.isclose(visible[:,0].mean(),row['target_in_frustum_fraction']) or not np.isclose((visible[:,0]&visible[:,1]).mean(),row['target_and_wrist_in_frustum_fraction']):
                raise ValueError('Simulation framing receipt differs from measured trajectory')
            valid = (np.arange(count) >= 4) & (np.arange(count) + 12 < count)
            selected = np.flatnonzero(valid)
            if not len(selected):raise ValueError("No complete simulation windows")
            actions = np.zeros((count,10,47), dtype=np.float32)
            states = np.zeros((count,4,77), dtype=np.float32)
            for i in selected:
                future = frames[i+1:i+11]
                actions[i,:,:45] = human_features(poses[future], poses[frames[i]], .44)
                actions[i,:,45:] = data["touch"][future]
            history_frames = frames[selected[:,None] + np.arange(-3,1)]
            states[selected,:,:75] = human_history(poses, stamps, history_frames, .44)
            states[selected,:,75:] = data["touch"][history_frames]
            if not np.isfinite(actions).all() or not np.isfinite(states).all():raise ValueError("Nonfinite simulation labels")
            rot = actions[selected,:,15:45].reshape(-1,5,6).reshape(-1,5,2,3).swapaxes(-1,-2)
            if np.max(abs(rot.swapaxes(-1,-2) @ rot - np.eye(2))) >= 1e-4:raise ValueError("Invalid rot6d orthogonality")
            video = dataset / f"videos/chunk-000/observation.images.cam_head/episode_{index:06d}.mp4"

            def images():
                for rgb in data["rgb"]:
                    buffer=io.BytesIO();Image.fromarray(rgb).save(buffer, format="PNG")
                    yield {"bytes":buffer.getvalue()}

            encode_video(images(), video)
            traj_id = f"q1/sim_{row.get('task', 'press_button')}_seed_{row['seed']:06d}"
            table = pa.table({"action":pa.array(actions.tolist(),type=pa.list_(pa.list_(pa.float32(),47),10)),
                "observation.state":pa.array(states.tolist(),type=pa.list_(pa.list_(pa.float32(),77),4)),
                "anchor_quaternion_wxyz":pa.array(poses[frames,0,3:].tolist(),type=pa.list_(pa.float32(),4)),
                "action.is_pad":pa.array(np.repeat((~valid)[:,None],10,axis=1).tolist(),type=pa.list_(pa.bool_(),10)),
                "timestamp":frames/50,"frame_index":np.arange(count),"episode_index":np.full(count,index),
                "index":np.arange(sum(r['length'] for r in episodes),sum(r['length'] for r in episodes)+count),
                "task_index":np.full(count,task),"valid_for_training":valid,"traj_id":[traj_id]*count})
            parquet=dataset/f"data/chunk-000/episode_{index:06d}.parquet"
            parquet.parent.mkdir(parents=True,exist_ok=True);pq.write_table(table,parquet,compression="zstd")
            trajectory=output/"sim_50hz_store"/(traj_id+".npz")
            trajectory.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(source,trajectory)
            saved={"episode_index":index,"length":count,"valid_samples":int(valid.sum()),"task_index":task,
                "seed":row["seed"],"split":row["split"],"traj_id":traj_id,"source_sha256":sha256(episode),
                "outputs":{str(p.relative_to(output)):sha256(p) for p in [video,parquet,trajectory]}}
            write_json(receipt,saved);episodes.append(saved)
    if not episodes:raise ValueError("No accepted simulation episodes")
    for row in episodes:
        if row['split']!='train':continue
        table=pq.read_table(dataset/f"data/chunk-000/episode_{row['episode_index']:06d}.parquet")
        mask=table['valid_for_training'].to_numpy()
        action_values.append(np.asarray(table['action'].to_pylist(),dtype=np.float32)[mask].reshape(-1,47))
        state_values.append(np.asarray(table['observation.state'].to_pylist(),dtype=np.float32)[mask].reshape(-1,77))
    stats={}
    for key,values in [('action',action_values),('observation.state',state_values)]:
        value=np.concatenate(values);stats[key]={'count':len(value),'mean':value.mean(0).tolist(),'std':value.std(0).tolist()}
    meta=dataset/'meta';meta.mkdir(parents=True,exist_ok=True)
    for name,rows in [('episodes.jsonl',episodes),('tasks.jsonl',[{'task_index':i,'task':t} for t,i in tasks.items()])]:
        (meta/name).write_text(''.join(json.dumps(r)+'\n' for r in rows))
    splits={s:[r['episode_index'] for r in episodes if r['split']==s] for s in ['train','val','test']}
    write_json(meta/'splits.json',splits);write_json(meta/'stats.json',stats)
    tasks_present = sorted({json.loads((raw_root / "episodes" / f"seed_{r['seed']:06d}" / "episode.json").read_text()).get("task", "press_button") for r in episodes})
    source={'schema':'q1-sim-button-five-point-v1','embodiment_tag':'bimanual_stub','label_provenance':'mujoco_simulated_robot_state',
        'coordinate_frame':'full t0 pelvis 3D; explicit anchor quaternion in every row','leg_length_m':.44,
        'end_effector_supervision':'measured simulated contact force >= 0.5 N; no finger joints',
        'tasks_present':tasks_present,
        'scope':'Q1 tabletop heading-locked expert with visual frustum framing gate; no finger joints'}
    write_json(dataset/'source.json',source);write_json(meta/'embodiment.json',source)
    write_json(meta/'modality.json',{'action':{'five_point':[0,45],'touch':[45,47]},'action_horizon':10,
        'state':{'pose':[0,45],'linear_velocity':[45,60],'angular_velocity':[60,75],'touch':[75,77]},'history_frames':4})
    write_json(meta/'info.json',{**source,'codebase_version':'v2.1','fps':10,'world_model':'Cosmos-Predict2.5-2B',
        'total_episodes':len(episodes),'total_frames':sum(r['length'] for r in episodes),'total_tasks':len(tasks),
        'features':{'action':{'dtype':'float32','shape':[10,47]},'observation.state':{'dtype':'float32','shape':[4,77]},
                    'observation.images.cam_head':{'dtype':'video','shape':[3,224,224]}}})
    report={'episodes':len(episodes),'valid_windows':sum(r['valid_samples'] for r in episodes),
        'split_episodes':{s:len(v) for s,v in splits.items()},'tasks_present':tasks_present,'ready':False,
        'pending':['video/text caches','validate-sim','20 Rerun visual inspections']}
    write_json(output/'qa_reports/sim_import.json',report)
    return report


def validate_sim(output):
    import torch
    from safetensors import safe_open
    from torch.utils.data import DataLoader
    from .datasets import WAMDataset
    from .geometry import matrices, rotation6d

    output = Path(output)
    dataset = output / DATASET
    video = json.loads((output / "qa_reports/cache_video.json").read_text())
    text = json.loads((output / "prompt_bank.json").read_text())
    if not video.get("complete") or not text.get("complete"):
        raise ValueError("Incomplete sim caches")
    if sha256(output / "prompt_bank_bf16.safetensors") != text["sha256"]:
        raise ValueError("Changed sim prompt bank")
    splits = json.loads((dataset / "meta/splits.json").read_text())
    memberships = [s for values in splits.values() for s in values]
    if len(memberships) != len(set(memberships)):
        raise ValueError("Sim episode split leakage")
    episodes = {r["episode_index"]: r for r in map(json.loads, (dataset / "meta/episodes.jsonl").read_text().splitlines())}
    samples = 0
    max_pos_err, max_rot_err = 0.0, 0.0
    for row in video["episodes"]:
        path = output / row["path"]
        if sha256(path) != row["sha256"]:
            raise ValueError(f"Changed sim latent cache: {path}")
        ep = episodes[row["episode_index"]]
        for name, digest in ep["outputs"].items():
            if sha256(output / name) != digest:
                raise ValueError(f"Changed sim paired source: {name}")
        with safe_open(str(path), framework="pt", device="cpu") as src:
            frames = src.get_tensor("frame_index").numpy()
            if (frames < 4).any() or (frames + 12 >= ep["length"]).any():
                raise ValueError("Invalid causal sim visual window")
            if src.get_slice("latents").get_shape() != [len(frames), 16, 5, 28, 28]:
                raise ValueError("Wrong sim latent dimensions")
            for start in range(0, len(frames), 16):
                if not torch.isfinite(src.get_slice("latents")[start:start + 16]).all():
                    raise ValueError("Nonfinite sim latent")
        table = pq.read_table(dataset / f"data/chunk-000/episode_{row['episode_index']:06d}.parquet")
        actions = np.asarray(table["action"].to_pylist(), dtype=np.float32)[frames]
        states = np.asarray(table["observation.state"].to_pylist(), dtype=np.float32)[frames]
        if not np.isfinite(actions).all() or not np.isfinite(states).all():
            raise ValueError("Nonfinite sim actions/states")
        if ((actions[..., 45:] < 0) | (actions[..., 45:] > 1)).any():
            raise ValueError("Sim touch target outside [0, 1]")
        rot = actions[..., 15:45].reshape(-1, 5, 2, 3)
        if not np.allclose(np.linalg.norm(rot, axis=-1), 1, atol=1e-4) or not np.allclose((rot[:, :, 0] * rot[:, :, 1]).sum(axis=-1), 0, atol=1e-4):
            raise ValueError("Nonorthogonal sim rot6d")
        # Roundtrip check on first sampled window of first 10 episodes
        if row["episode_index"] < 10 and len(frames):
            f0 = int(frames[0])
            with np.load(output / "sim_50hz_store" / (ep["traj_id"] + ".npz"), allow_pickle=False) as raw_npz:
                poses = raw_npz["poses_world"]
                rgb_idx = raw_npz["rgb_frame_index"]
                anchor = poses[rgb_idx[f0]]
                future = poses[rgb_idx[f0 + 1:f0 + 11]]
                r0 = matrices(anchor[None, 0, 3:])[0]
                act = actions[0, :, :45]
                pos_local = act[:, :15].reshape(10, 5, 3).copy()
                pos_local[:, :3] *= 0.44
                pos_world = pos_local @ r0.T + anchor[0, :3]
                rot_local = rotation6d(act[:, 15:].reshape(10, 5, 6))
                rot_world = r0[None, None] @ rot_local
                gt_rot = matrices(future[:, :, 3:].reshape(-1, 4)).reshape(10, 5, 3, 3)
                max_pos_err = max(max_pos_err, float(np.linalg.norm(pos_world - future[:, :, :3], axis=-1).max()))
                tr = np.clip((np.einsum("...ij,...ij->...", rot_world, gt_rot) - 1.0) / 2.0, -1.0, 1.0)
                max_rot_err = max(max_rot_err, float(np.rad2deg(np.arccos(tr)).max()))
        samples += len(frames)
    if max_pos_err >= 1e-3 or max_rot_err >= 0.1:
        raise ValueError(f"Sim roundtrip error exceeded gate: {max_pos_err} m, {max_rot_err} deg")
    reader = WAMDataset(output, "train", dataset_name="q1_stub")
    batch = next(iter(DataLoader(reader, batch_size=1)))
    expected = {"video_latents": (1, 16, 5, 28, 28), "text_embeddings": (1, 512, 100352),
                "proprio": (1, 4, 77), "actions": (1, 10, 47)}
    for key, shape in expected.items():
        if batch[key].shape != shape or not torch.isfinite(batch[key]).all():
            raise ValueError(f"Invalid sim WAM batch: {key} got {tuple(batch[key].shape)}")
    head = torch.nn.Linear(16 + 4 * 77, 470)
    feat = torch.cat([batch["video_latents"].float().mean(dim=(2, 3, 4)), batch["proprio"].float().flatten(1)], dim=1)
    loss = (head(feat) - batch["actions"].flatten(1)).square().mean()
    loss.backward()
    if not torch.isfinite(head.weight.grad).all():
        raise ValueError("Invalid sim WAM gradients")
    split_samples = {s: len(WAMDataset(output, s, dataset_name="q1_stub")) for s in ("train", "val", "test") if splits.get(s)}
    report = {
        "ready": True, "backbone": "Cosmos-Predict2.5-2B", "episodes": len(episodes),
        "cached_samples": samples, "split_samples": split_samples,
        "maximum_position_roundtrip_m": max_pos_err, "maximum_rotation_roundtrip_degrees": max_rot_err,
        "dataset_source_sha256": sha256(dataset / "source.json"),
        "episodes_sha256": sha256(dataset / "meta/episodes.jsonl"),
        "video_cache_report_sha256": sha256(output / "qa_reports/cache_video.json"),
        "text_cache_sha256": text["sha256"],
        "batch_shapes": {k: list(v) for k, v in expected.items()}, "finite_gradient": True,
    }
    write_json(output / "qa_reports/sim_wam_training_ready.json", report)
    return report
